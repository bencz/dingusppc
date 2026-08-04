/*
DingusPPC - The Experimental PowerPC Macintosh emulator
Copyright (C) 2018-26 The DingusPPC Development Team
          (See CREDITS.MD for more details)

(You may also contact divingkxt or powermax2286 on Discord)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/** @file Where the pieces meet: lookup, translation and the block executing
    counterpart of ppc_exec_inner.
 */

#include "../ppcjit.h"
#include "../ppccodecache.h"
#include "../ppcemu.h"
#include "../ppcmmu.h"
#include "backend.h"
#include "jitcache.h"
#include "jitir.h"
#include "jitruntime.h"

#include <loguru.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

bool ppc_jit_enabled   = false;
bool ppc_jit_requested = false;

namespace dppc_jit {

std::unique_ptr<Backend> make_native_backend() {
#if defined(DPPC_JIT_X86_64)
    return make_x86_64_backend();
#elif defined(DPPC_JIT_AARCH64)
    return make_aarch64_backend();
#else
    return nullptr;
#endif
}

namespace {

std::unique_ptr<Backend> native_backend;
std::unique_ptr<Backend> threaded_backend;

/** The selected executor and its fallback policy, for diagnostics */
std::string backend_label;

/** Blocks the code cache dropped but nobody has freed yet.

    Invalidation fires from inside a guest instruction, which means it can
    reach the very block that instruction belongs to. Freeing it there would
    pull the ground out from under the block still walking its own body, so
    the release only unlinks and the memory goes back at the next block
    boundary, where nothing is executing */
std::vector<JitBlock*> pending_free;

/** Scratch, reused across translations to keep the vector inside it warm */
IRBlock scratch_ir;

unsigned native_compiles   = 0;
unsigned threaded_compiles = 0;

uint64_t va_verify_checks      = 0;
uint64_t va_verify_successes   = 0;
uint64_t va_verify_resolvers   = 0;
uint64_t va_verify_failures    = 0;
uint64_t va_verify_exceptions  = 0;
uint32_t va_verify_reports     = 0;

uint64_t va_translate_checks       = 0;
uint64_t va_translate_primary_hits = 0;
uint64_t va_translate_misses       = 0;

/** Interpreted block entries a candidate collects before the emitter is asked.

    A booting system runs megabytes of code exactly once: loaders, linkers,
    initialisation that never comes back. Emitting all of it cost more than
    running it, and the churn was worse than the cost: every block thrown
    away by an invalidation had bought chain bindings, registry entries and
    code pool bytes with it. Below the threshold the ordinary interpreter
    runs the instructions; crossing it buys emission once.

    Zero turns the gate off and every block compiles native on first entry,
    which is what the tests use to hold the emitter to full coverage.
    DPPC_JIT_HEAT overrides it for bisection either way */
uint32_t heat_threshold = 8;

/** A bounded hotness map, deliberately separate from the code cache.

    Cold code must not need a JitBlock allocation just to remember that it
    ran once. A direct-mapped table keeps that state to a fixed amount of BSS
    and makes eviction harmless: a collision can delay compilation, never
    change guest behaviour. Count zero marks an empty/cooling entry, so key
    zero needs no special case. */
struct HeatEntry {
    uint64_t key;
    uint32_t count;
    uint32_t epoch;
};

constexpr size_t HEAT_TABLE_SIZE = 1u << 16;
constexpr size_t HEAT_TABLE_MASK = HEAT_TABLE_SIZE - 1;
HeatEntry heat_table[HEAT_TABLE_SIZE] = {};
uint32_t heat_epoch = 1;

inline size_t heat_index(uint64_t key) {
    uint32_t mixed = uint32_t(key) ^ uint32_t(key >> 32);
    mixed ^= mixed >> 16;
    mixed *= 0x7FEB352Du;
    mixed ^= mixed >> 15;
    return size_t(mixed) & HEAT_TABLE_MASK;
}

bool heat_ready(uint64_t key) {
    if (!heat_threshold) {
        return true;
    }

    HeatEntry& entry = heat_table[heat_index(key)];
    if (entry.epoch != heat_epoch || !entry.count || entry.key != key) {
        entry.key   = key;
        entry.count = 1;
        entry.epoch = heat_epoch;
    } else if (entry.count < heat_threshold) {
        entry.count++;
    }

    if (entry.count < heat_threshold) {
        return false;
    }

    // Space retries out too: an untranslatable or emitter-declined block
    // gets another attempt only after a fresh interval of interpreted runs.
    entry.count = 0;
    return true;
}

void heat_clear() {
    // Epoch invalidation makes the common flush O(1). Only the impossible in
    // practice 32-bit wrap has to touch the table before epoch one is reused.
    if (++heat_epoch == 0) {
        for (HeatEntry& entry : heat_table) {
            entry.epoch = 0;
        }
        heat_epoch = 1;
    }
}

/** What the outer loop is doing, so rt_dispatch can apply the same stopping
    rule from inside generated code. Set once per ppc_jit_exec_inner and read
    between blocks */
JitExecType run_type = JitExecType::run;
uint32_t    run_goal = 0;

/** Blocks whose chain_in is nonempty, so unbinding everything walks only
    them instead of every block there is */
JitBlock* chained_head = nullptr;

/** Bound slots, purely a statistic. Entries live in the slots themselves,
    so the registry cannot grow past the slots emitted code holds and there
    is no cap to police. The capped pooled registry this replaces tore every
    chain in the machine down about once a second during the Cheetah storm,
    address predicted slots rebinding on every miss being what filled it */
size_t chain_refs = 0;

/** Binding is off during `until` runs, where every block entry has to be
    observed for the goal, and under tracing, which wants every entry too */
bool chain_allowed = false;

void unlink_chained_block(JitBlock* blk) {
    if (blk->chained_prev) {
        blk->chained_prev->chained_next = blk->chained_next;
    } else {
        chained_head = blk->chained_next;
    }
    if (blk->chained_next) {
        blk->chained_next->chained_prev = blk->chained_prev;
    }
    blk->chained_prev = nullptr;
    blk->chained_next = nullptr;
}

/** Detaches a ref from its target's incoming list without touching the slot,
    for a way that is about to be rewritten with a fresh binding anyway */
inline bool chain_ref_jumps_direct(const ChainRef* ref) {
    return *ref->code != ref->resolver;
}

void unlink_chain_ref(ChainRef* ref) {
    JitBlock* blk = ref->target;
    const bool direct = chain_ref_jumps_direct(ref);
    if (ref->prev) {
        ref->prev->next = ref->next;
    } else {
        blk->chain_in = ref->next;
        if (!ref->next) {
            unlink_chained_block(blk);
        }
    }
    if (ref->next) {
        ref->next->prev = ref->prev;
    }
    ref->target = nullptr;
    if (direct) {
        chain_refs--;
    }
}

/** Tracks a slot's target on its incoming list. A direct entry becomes a real
    bound jump immediately after this returns; an observed entry deliberately
    leaves its code cell on the resolver and only caches target identity. */
void track_chain(JitBlock* blk, ChainRef* ref, void** code, const void* resolver,
                 uint64_t* pred, bool direct) {
    if (ref->target) {
        unlink_chain_ref(ref);
    }
    ref->code     = code;
    ref->resolver = resolver;
    ref->pred     = pred;
    ref->target   = blk;
    ref->prev     = nullptr;
    ref->next     = blk->chain_in;
    if (ref->next) {
        ref->next->prev = ref;
    } else {
        blk->chained_prev = nullptr;
        blk->chained_next = chained_head;
        if (chained_head) {
            chained_head->chained_prev = blk;
        }
        chained_head = blk;
    }
    blk->chain_in = ref;
    if (direct) {
        chain_refs++;
    }
}

/** Points a slot's registry entry at a new target. A rebinding entry moves
    off the old target's list first, so nothing is allocated and nothing is
    left behind, however hard a site churns */
void bind_chain(JitBlock* blk, ChainRef* ref, void** code, const void* resolver,
                uint64_t* pred) {
    track_chain(blk, ref, code, resolver, pred, true);
}

/** Remembers a resolver observation without allowing generated code to skip
    the resolver on its next entry. */
void cache_chain_target(JitBlock* blk, ChainRef* ref, void** code,
                        const void* resolver, uint64_t* pred = nullptr) {
    track_chain(blk, ref, code, resolver, pred, false);
}

/** Sends every slot aimed at the block back to resolving */
void unbind_chains_to(JitBlock* blk) {
    ChainRef* ref = blk->chain_in;
    if (!ref) {
        return;
    }
    do {
        if (chain_ref_jumps_direct(ref)) {
            chain_refs--;
        }
        *ref->code = const_cast<void*>(ref->resolver);
        if (ref->pred) {
            *ref->pred = 1;
        }
        ref->target = nullptr;
        ref = ref->next;
    } while (ref);
    blk->chain_in = nullptr;
    unlink_chained_block(blk);
}

/** Detaches every binding owned by a block that is about to die. Incoming
    tracking alone is not enough: a live target may otherwise retain a
    ChainRef stored in the dead block's slot area, and a later invalidation
    would write through that dangling entry. */
void unbind_chains_from(JitBlock* blk) {
    for (ChainRef* ref = blk->chain_out; ref; ref = ref->owner_next) {
        if (ref->target) {
            unlink_chain_ref(ref);
        }
    }
    blk->chain_out = nullptr;
}

void unbind_all_chains() {
    for (JitBlock* blk = chained_head; blk;) {
        JitBlock* next_blk = blk->chained_next;
        for (ChainRef* ref = blk->chain_in; ref; ref = ref->next) {
            *ref->code = const_cast<void*>(ref->resolver);
            if (ref->pred) {
                *ref->pred = 1;
            }
            ref->target = nullptr;
        }
        blk->chain_in     = nullptr;
        blk->chained_prev = nullptr;
        blk->chained_next = nullptr;
        blk = next_blk;
    }
    chained_head = nullptr;
    chain_refs   = 0;
}

bool validate_chain_graph(std::string* error, bool* can_unbind = nullptr) {
    if (can_unbind) {
        *can_unbind = false;
    }
    auto fail = [error](const char* why) {
        if (error) {
            *error = why;
        }
        return false;
    };

    const std::vector<JitBlock*> live_blocks = cache_blocks();
    std::unordered_set<JitBlock*> live;
    live.reserve(live_blocks.size());
    for (JitBlock* blk : live_blocks) {
        if (!blk) {
            return fail("the code cache contains a null block");
        }
        if (!live.insert(blk).second) {
            return fail("the code cache contains the same block twice");
        }
    }

    // First establish ownership. This catches a cycle in a source list and
    // one slot being claimed by two source blocks before incoming links are
    // trusted.
    std::unordered_set<ChainRef*> owned_refs;
    for (JitBlock* owner : live_blocks) {
        for (ChainRef* ref = owner->chain_out; ref; ref = ref->owner_next) {
            if (!owned_refs.insert(ref).second) {
                return fail("a chain entry is duplicated or cyclic in source ownership");
            }
            if (ref->target && !live.count(ref->target)) {
                return fail("a source chain entry points at a dead target block");
            }
        }
    }

    std::unordered_set<JitBlock*> listed_blocks;
    std::unordered_set<ChainRef*> incoming_refs;
    JitBlock* expected_prev = nullptr;
    for (JitBlock* blk = chained_head; blk; blk = blk->chained_next) {
        if (!listed_blocks.insert(blk).second) {
            return fail("the chained-block registry is duplicated or cyclic");
        }
        if (!live.count(blk)) {
            return fail("the chained-block registry contains a dead block");
        }
        if (blk->chained_prev != expected_prev) {
            return fail("the chained-block previous link is inconsistent");
        }
        if (!blk->chain_in) {
            return fail("a chained-block registry entry has no incoming chains");
        }

        ChainRef* expected_ref_prev = nullptr;
        for (ChainRef* ref = blk->chain_in; ref; ref = ref->next) {
            if (!incoming_refs.insert(ref).second) {
                return fail("an incoming chain entry is duplicated or cyclic");
            }
            if (!owned_refs.count(ref)) {
                return fail("an incoming chain entry has no live source owner");
            }
            if (ref->target != blk) {
                return fail("an incoming chain entry names a different target");
            }
            if (ref->prev != expected_ref_prev) {
                return fail("an incoming chain previous link is inconsistent");
            }
            if (!ref->code || !ref->resolver) {
                return fail("a tracked chain entry has an incomplete code cell");
            }
            expected_ref_prev = ref;
        }
        expected_prev = blk;
    }

    for (JitBlock* blk : live_blocks) {
        const bool listed = listed_blocks.count(blk) != 0;
        if ((blk->chain_in != nullptr) != listed) {
            return fail("a block's incoming-list state disagrees with the registry");
        }
    }
    for (ChainRef* ref : owned_refs) {
        const bool incoming = incoming_refs.count(ref) != 0;
        if ((ref->target != nullptr) != incoming) {
            return fail("a source chain's target state disagrees with incoming lists");
        }
    }

    // All lists are finite, mutually consistent and backed by live owners at
    // this point. A semantic failure below can therefore be contained by
    // walking them once more and restoring every resolver.
    if (can_unbind) {
        *can_unbind = true;
    }

    size_t direct_refs = 0;
    for (ChainRef* ref : incoming_refs) {
        JitBlock* blk = ref->target;
        if (ref->pred) {
            const uint64_t prediction = *ref->pred;
            if (prediction > UINT32_MAX || prediction == 1 ||
                (prediction & 3) != 0) {
                return fail("a tracked virtual chain has an invalid prediction");
            }
            if ((uint32_t(prediction) & ~PPC_PAGE_MASK) !=
                (blk->phys_addr & ~PPC_PAGE_MASK)) {
                return fail("a virtual chain prediction and target offset disagree");
            }
        }
        if (*ref->code != ref->resolver) {
            if (*ref->code != blk->code) {
                return fail("a direct chain points somewhere other than its target code");
            }
            direct_refs++;
        }
    }
    if (direct_refs != chain_refs) {
        return fail("the direct-chain statistic disagrees with the graph");
    }

    if (error) {
        error->clear();
    }
    return true;
}

/** Registers a block's invalidation ranges, the cross page walk through's
    second one included */
void register_ranges(JitBlock* blk) {
    ppc_code_cache_add(blk->phys_addr, blk->byte_size,
                       static_cast<CodeBlockHandle>(blk));
    if (blk->second_size) {
        ppc_code_cache_add(blk->second_phys, blk->second_size,
                           static_cast<CodeBlockHandle>(blk));
    }
}

void on_block_released(CodeBlockHandle handle) {
    JitBlock* blk = static_cast<JitBlock*>(handle);
    if (blk->second_size) [[unlikely]] {
        // both registrations die with the block. Whichever one the
        // invalidation walk is standing on only gets tombstoned, which the
        // walk survives; the other would dangle without this
        ppc_code_cache_remove(blk->phys_addr, handle);
        ppc_code_cache_remove(blk->second_phys, handle);
    }
    unbind_chains_to(blk);
    unbind_chains_from(blk);
    cache_forget(blk);
    pending_free.push_back(blk);
}

void drain_pending_free() {
    if (pending_free.empty()) [[likely]] {
        return;
    }
    for (JitBlock* blk : pending_free) {
        blk->owner->release(blk);
    }
    pending_free.clear();
}

/** Drops every translation and gives the backends their pools back.

    Safe to call between two blocks and nowhere else. Generated code that is
    on the stack at that moment lives in the shared stubs, which sit below the
    code memory floor and survive this */
void flush_everything() {
    ppc_code_cache_invalidate_all(); // sends every block through on_block_released
    drain_pending_free();
    cache_clear();
    heat_clear();

    // every release above already unbound its chains and emptied the chained
    // list; the entries themselves live in the slot area the backends are
    // about to take back wholesale
    chained_head = nullptr;
    chain_refs   = 0;

    if (native_backend) {
        native_backend->release_all();
    }
    if (threaded_backend) {
        threaded_backend->release_all();
    }
}

/** Finds the block covering virt_addr under the current mode, translating it
    once its interpreter entry count is hot enough. Returns nullptr when the
    instruction there has to go back to the interpreter.

    allow_flush says the caller sits below the code memory floor, which the
    dispatch stub does and a resolver thunk does not: a thunk lives in the
    block that emitted it, a flush reclaims that memory, and the very next
    compile writes new code over the return address the resolver call left
    on the stack. A resolver therefore passes false and leaves the block
    untranslated rather than flushing; its exit falls back to the stub, and
    the stub's own lookup does the flushing on safe ground.

    count_heat is true only at the outer execution loop, immediately before
    an interpreted candidate block really runs. Resolver and dispatch probes
    do not count, otherwise one cold entry can be charged several times. A
    bounded cold span returns here at every guest control-flow boundary, so a
    tight loop still contributes exactly one entry per iteration.

    When interp_code is supplied, it receives the translated host instruction
    pointer. The outer loop reuses it if this lookup leaves the instruction to
    the interpreter, avoiding a second translation of the same address.

    Can unwind out of mmu_translate_imem on an instruction fetch fault, which
    is fine: no generated frame is on the stack at this point */
JitBlock* find_or_translate(uint32_t virt_addr, bool allow_flush, bool count_heat,
                            const uint8_t** interp_code = nullptr,
                            bool* cold_fallback = nullptr) {
    if (interp_code) {
        *interp_code = nullptr;
    }
    if (cold_fallback) {
        *cold_fallback = false;
    }
#if SUPPORTS_PPC_LITTLE_ENDIAN_MODE
    if (ppc_state.is_LE) [[unlikely]] {
        // little endian munges addresses, so walking a host pointer forward
        // no longer lands on the next instruction. Not worth an axis in the
        // IR, the interpreter handles it
        return nullptr;
    }
#endif

    const uint32_t mode = ppc_jit_mode();

    uint32_t phys_addr = 0;
    const uint8_t* code = mmu_translate_imem(virt_addr, &phys_addr);
    if (interp_code) {
        *interp_code = code;
    }

    if (JitBlock* blk = cache_lookup(phys_addr, mode)) {
        return blk;
    }

    // Automatic mode has a native backend and leaves cold instructions to
    // the interpreter. Explicit threaded mode has no native backend and is
    // compiled immediately, because its purpose is differential testing.
    if (native_backend && heat_threshold &&
        (!count_heat || !heat_ready(block_key(phys_addr, mode)))) {
        if (cold_fallback && count_heat) {
            *cold_fallback = true;
        }
        return nullptr;
    }

    if (!translate_block(virt_addr, phys_addr, code, mode, scratch_ir)) {
        return nullptr;
    }

    Backend* owner = native_backend ? native_backend.get() : threaded_backend.get();
    JitBlock* blk  = owner->compile(scratch_ir);

    if (!blk && native_backend && owner->wants_flush()) {
        if (!allow_flush) {
            // A resolver is on the stack with its return address inside the
            // memory a flush reclaims. The dispatch stub retries from safe
            // ground; with a heat gate, the outer loop does so after it has
            // counted the actual interpreted entry.
            return nullptr;
        }
        // The pool filled up. Everything translated so far goes, and this
        // block gets one more try on the empty pool. Nothing is executing
        // here, so reclaimed code cannot be under anyone's feet.
        LOG_F(INFO, "JIT: code memory exhausted, flushing %zu blocks", cache_size());
        flush_everything();
        blk = owner->compile(scratch_ir);
    }

    if (!blk) {
        return nullptr; // the ordinary interpreter is the automatic fallback
    }

    if (native_backend) {
        native_compiles++;
    } else {
        threaded_compiles++;
    }
    blk->owner = owner;

    cache_insert(blk);
    register_ranges(blk);
    return blk;
}

/** Ring of the guest addresses generated code most recently entered, for
    working out how the guest got somewhere it should not be.

    Off unless DPPC_JIT_TRACE is set, because it costs a store on the hottest
    path there is. Dumped by the fatal handler, since the way this tends to
    show up is an abort deep in the MMU with no idea what jumped there */
constexpr size_t TRACE_SIZE = 1u << 9;

struct TraceEntry {
    uint32_t pc;
    uint32_t mode;
    uint16_t insns;
    uint8_t  end_reason;
    bool     native;
    uint32_t words[8]; // what the guest actually had there, up to eight
};

TraceEntry trace_ring[TRACE_SIZE];
size_t     trace_next    = 0;
bool       trace_enabled = false;

inline void trace_block(const JitBlock* blk) {
    if (!trace_enabled) [[likely]] {
        return;
    }
    TraceEntry& e = trace_ring[trace_next++ & (TRACE_SIZE - 1)];
    e.pc         = ppc_state.pc;
    e.mode       = blk ? blk->mode : ppc_jit_mode();
    e.insns      = blk ? uint16_t(blk->insn_count) : 0;
    e.end_reason = blk ? blk->end_reason : 0xFF;
    e.native     = blk && blk->code;

    // read through the same fetch path the block was translated through, so
    // what shows up is what the guest has there now, not what it had then
    for (unsigned i = 0; i < 8; i++) {
        e.words[i] = 0;
    }
    const unsigned want = e.insns < 8 ? e.insns : 8;
    for (unsigned i = 0; i < want; i++) {
        uint32_t pa = 0;
        if (!mmu_translate_dbg(e.pc + i * 4, pa)) {
            break;
        }
        e.words[i] = uint32_t(mem_read_dbg(e.pc + i * 4, 4));
    }
}

void trace_dump() {
    if (!trace_enabled) {
        return;
    }
    const size_t count = trace_next < TRACE_SIZE ? trace_next : TRACE_SIZE;
    LOG_F(ERROR, "last %zu blocks the JIT entered, oldest first:", count);
    for (size_t i = 0; i < count; i++) {
        const TraceEntry& e = trace_ring[(trace_next - count + i) & (TRACE_SIZE - 1)];
        char words[8 * 9 + 1] = {0};
        for (unsigned w = 0; w < 8 && w < e.insns; w++) {
            snprintf(words + w * 9, 10, "%08X ", e.words[w]);
        }
        LOG_F(ERROR, "  pc 0x%08X  mode 0x%02X  %2u insns  %-14s %-8s  %s",
              e.pc, e.mode, e.insns,
              e.end_reason == 0xFF ? "interpreted"
                                   : block_end_name(BlockEnd(e.end_reason)),
              e.native ? "native" : "threaded", words);
    }
}

/** True when an `until` run would have to observe the PC somewhere inside
    this block.

    The interpreter compares after every instruction, so a goal in the middle
    of a block is a stop it would make. Rather than shaping blocks around
    whatever goal was asked for, such a block is stepped through on the
    interpreter */
bool goal_splits_block(const JitBlock* blk, uint32_t entry_pc) {
    if (run_type != JitExecType::until) {
        return false;
    }
    // conservative both ways: gaps a walk through skipped count as covered
    if (run_goal > entry_pc && run_goal < entry_pc + blk->byte_size) {
        return true;
    }
    // the callee region of a cross page walk through, by its guest address;
    // a block with one always ends inside it, so its start is end_off minus
    // its size, and modular arithmetic keeps a backward call honest
    if (blk->second_size) {
        const uint32_t s = entry_pc + uint32_t(blk->end_off) - blk->second_size;
        return run_goal - s < blk->second_size;
    }
    return false;
}

/** One instruction on the interpreter, for whatever no backend took.
    pc_real is normally the translation already obtained by find_or_translate;
    little-endian mode reaches here without one and translates on demand.
    Leaves the PC and exec_flags exactly as a block would */
uint32_t interpret_one(const uint8_t* pc_real,
                       uint64_t& deadline = g_icycles_max) {
    if (!pc_real) {
        pc_real = mmu_translate_imem(ppc_state.pc);
    }
    uint32_t opcode  = ppc_read_instruction(pc_real);

    ppc_main_opcode(ppc_opcode_grabber, opcode);
    ppc_account_cycles(1, deadline);

    if (!exec_flags) {
        ppc_state.pc += 4;
    }
    return opcode;
}

/** Runs one cold candidate block through the ordinary instruction helpers.

    Returning to find_or_translate after every instruction made cache lookup,
    hotness accounting and dispatch cost several times more than executing the
    instruction. The translator has not inspected cold code yet, so this uses
    conservative boundaries and the same hard instruction budget. It only
    serves the below-threshold path: an actual translator or emitter decline
    still executes exactly one instruction before the next lookup. */
void interpret_cold_span(const uint8_t* pc_real) {
    uint64_t deadline = g_icycles_max;
    for (uint32_t i = 0; i < jit_max_block_insns; i++) {
        const uint32_t opcode = interpret_one(pc_real, deadline);

        if (!power_on || exec_flags || jit_fallback_ends_span(opcode) ||
            (run_type == JitExecType::until && ppc_state.pc == run_goal) ||
            i + 1 == jit_max_block_insns ||
            (ppc_state.pc & ~PPC_PAGE_MASK) == 0) {
            return;
        }

        pc_real += 4;
    }
}

/** rt_dispatch proper, minus the guarantee that nothing unwinds out of it */
const void* dispatch_body(uint32_t retired) {
    // the block already left ppc_state.pc on the next guest instruction,
    // which is what rt_account_cycles needs to be true before it runs
    rt_account_cycles(retired);

    if (!power_on) [[unlikely]] {
        return nullptr;
    }

    if (exec_flags) {
        // a helper branch, an exception or an rfi only redirected the PC,
        // and the loop outside would do nothing but copy the address and come
        // straight back in through the trampoline. Copying it here keeps the
        // frame standing, which matters because every bclr in a helper and
        // every interrupt used to pay that round trip. Sleep is the one flag
        // that needs the outer loop, for its event spin
        if (exec_flags & EXEF_SLEEP) [[unlikely]] {
            return nullptr;
        }
        ppc_state.pc = ppc_next_instruction_address;
        exec_flags   = 0;
    }

    // between two blocks nothing is executing, so anything invalidation
    // unlinked while the last one ran can go back now
    drain_pending_free();

    if (run_type == JitExecType::until && ppc_state.pc == run_goal) [[unlikely]] {
        return nullptr;
    }

    const uint32_t entry_pc = ppc_state.pc;
    JitBlock* blk = find_or_translate(entry_pc, true, false);
    trace_block(blk);

    if (!blk || !blk->code || goal_splits_block(blk, entry_pc)) [[unlikely]] {
        // a block the emitter declined runs as ordinary C++ and has no code
        // to jump to, so it ends the native run like everything else here.
        // The loop below looks it up again, which costs one hash probe on a
        // path that was already paying for an interpreter walk
        return nullptr;
    }

    return blk->code;
}

} // namespace

const void* rt_dispatch(uint32_t retired) noexcept {
    try {
        return dispatch_body(retired);
    } catch (PPCExcUnwind&) {
        // the fetch faulted or a timer raised. Either way the handler already
        // ran, so all that is left is getting off the generated frame
        return nullptr;
    }
}

const void* rt_chain_resolve(ChainSlot* slot) noexcept {
    try {
        // the chained exit settled the cycles and checked the timer before
        // coming here, so flags mean something translation raised last time
        // around; the dispatch stub knows what to do with all of it
        if (exec_flags || !power_on) [[unlikely]] {
            return nullptr;
        }

        drain_pending_free();

        if (run_type == JitExecType::until && ppc_state.pc == run_goal) [[unlikely]] {
            return nullptr;
        }

        const uint32_t entry_pc = ppc_state.pc;
        const void* resolver =
            slot->ref.target ? slot->ref.resolver : slot->code;
        JitBlock* blk = slot->ref.target;
        if (!blk) {
            blk = find_or_translate(entry_pc, false, false);
        }
        trace_block(blk);

        if (!blk || !blk->code || goal_splits_block(blk, entry_pc)) [[unlikely]] {
            return nullptr;
        }

        // the jump itself is valid either way; what `until` and tracing veto
        // is the binding, which would let later passes skip this observation
        if (chain_allowed) {
            // the cell still points at the resolver thunk, or this call
            // would not be happening
            bind_chain(blk, &slot->ref, &slot->code, resolver, nullptr);
            slot->code = blk->code;
        } else if (!slot->ref.target) {
            // Keep observing every entry, but do not repeat a full physical
            // lookup until invalidation clears this tracked weak reference.
            cache_chain_target(blk, &slot->ref, &slot->code, resolver);
        }
        return blk->code;
    } catch (PPCExcUnwind&) {
        // the fetch faulted; the handler already ran, leave through dispatch
        return nullptr;
    }
}

const void* rt_chain_resolve_va(ChainVaSlot* slot) noexcept {
    try {
        if (exec_flags || !power_on) [[unlikely]] {
            return nullptr;
        }

        drain_pending_free();

        if (run_type == JitExecType::until && ppc_state.pc == run_goal) [[unlikely]] {
            return nullptr;
        }

        const uint32_t entry_pc = ppc_state.pc;

        // An observed way always keeps code on the resolver, so getting here
        // is expected. Its address and translation generation have already
        // passed the emitted guards. A stale generation reaches the thunk
        // only when its inline ITLB revalidation failed, and must therefore
        // take the honest translation path below instead of trusting the old
        // physical target.
        JitBlock* blk = nullptr;
        if (slot->pred0 == entry_pc &&
            slot->gen0 == mmu_itrans_generation) {
            blk = slot->ref0.target;
        } else if (slot->pred1 == entry_pc &&
                   slot->gen1 == mmu_itrans_generation) {
            blk = slot->ref1.target;
        }
        if (!blk) {
            blk = find_or_translate(entry_pc, false, false);
        }
        trace_block(blk);

        if (!blk || !blk->code || goal_splits_block(blk, entry_pc)) [[unlikely]] {
            return nullptr;
        }

        // way choice: refresh the way already predicting this address, which
        // is how a stale generation gets restamped instead of duplicating the
        // prediction across ways; then a virgin way; then evict round robin.
        // An eviction moves the way's registry entry to the new target, so
        // nothing stale remains anywhere. Rebinding at all matters as much
        // as the ways do: a policy that only bound virgin slots left every
        // site dead after the first tlbie
        int way;
        if (slot->pred0 == entry_pc)      way = 0;
        else if (slot->pred1 == entry_pc) way = 1;
        else if (slot->pred0 == 1)        way = 0;
        else if (slot->pred1 == 1)        way = 1;
        else {
            way = int(slot->flip & 1);
            slot->flip ^= 1;
        }

        uint64_t* pred = way ? &slot->pred1 : &slot->pred0;
        uint64_t* gen  = way ? &slot->gen1  : &slot->gen0;
        void**    code = way ? &slot->code1 : &slot->code0;
        uint32_t* phys = way ? &slot->phys1 : &slot->phys0;
        ChainRef* ref  = way ? &slot->ref1  : &slot->ref0;

        if (chain_allowed && jit_va_binding) {

            // the storm case: the way already holds this very binding and
            // only the generation went stale. The translation was just
            // re-verified, so restamping it is enough, and the way's entry
            // already sits on the right list. The exit's inline probe
            // catches most of these; what reaches here is the probe missing
            // the primary ITLB, which the translation above has just
            // refilled
            if (*pred == entry_pc && *code == blk->code) {
                *gen = mmu_itrans_generation;
                return blk->code;
            }

            bind_chain(blk, ref, code, slot->resolver, pred);
            *pred = entry_pc;
            *gen  = mmu_itrans_generation;
            *phys = blk->phys_addr & PPC_PAGE_MASK;
            *code = blk->code;
        } else if (ref->target != blk || *pred != entry_pc ||
                   *gen != mmu_itrans_generation) {
            // Populate the ordinary address and generation guards, but keep
            // code pointing at the resolver. The next matching pass still
            // gets observed while avoiding MMU and block-cache lookup.
            cache_chain_target(blk, ref, code, slot->resolver, pred);
            *pred = entry_pc;
            *gen  = mmu_itrans_generation;
            *phys = blk->phys_addr & PPC_PAGE_MASK;
        }
        return blk->code;
    } catch (PPCExcUnwind&) {
        return nullptr;
    }
}

static const void* validate_chain_va(ChainVaSlot* slot, uint32_t way,
                                     uint32_t entry_pc,
                                     bool translate) noexcept {
    void** code = way ? &slot->code1 : &slot->code0;
    uint64_t* pred = way ? &slot->pred1 : &slot->pred0;
    uint64_t* gen = way ? &slot->gen1 : &slot->gen0;
    uint32_t* phys = way ? &slot->phys1 : &slot->phys0;
    ChainRef* ref = way ? &slot->ref1 : &slot->ref0;

    // An observed way deliberately still points at the resolver. Preserve
    // that path without charging it as a direct entry verification.
    if (*code == slot->resolver) {
        va_verify_resolvers++;
        return slot->resolver;
    }

    va_verify_checks++;

    auto reject = [&](const char* reason, uint32_t translated_phys,
                      JitBlock* cached) -> const void* {
        va_verify_failures++;
        if (va_verify_reports++ < 32) {
            const JitBlock* target = ref->target;
            LOG_F(ERROR,
                  "JIT VA verify failed (%s): pc=%08X way=%u pred=%08llX "
                  "gen=%llu/%llu phys=%08X translated=%08X target=%p "
                  "target_phys=%08X target_mode=%08X current_mode=%08X "
                  "cached=%p flags=%02X",
                  reason, entry_pc, way,
                  (unsigned long long)*pred,
                  (unsigned long long)*gen,
                  (unsigned long long)mmu_itrans_generation,
                  *phys, translated_phys, static_cast<const void*>(target),
                  target ? target->phys_addr : 0,
                  target ? target->mode : 0, ppc_jit_mode(),
                  static_cast<void*>(cached), unsigned(exec_flags));
        }

        // The generated frame owns this slot but no code is being reclaimed.
        // Detaching one bad way and poisoning its prediction is safe; dispatch
        // performs the authoritative retry from ppc_state.pc.
        if (ref->target) {
            unlink_chain_ref(ref);
        }
        *code = const_cast<void*>(slot->resolver);
        *pred = 1;
        *gen = 0;
        *phys = 0;
        return nullptr;
    };

    if (!power_on || exec_flags) {
        return reject("pending execution state", 0, nullptr);
    }
    if (*pred != entry_pc) {
        return reject("prediction changed", 0, nullptr);
    }
    if (*gen != mmu_itrans_generation) {
        return reject("translation generation changed", 0, nullptr);
    }
    if (!ref->target) {
        return reject("direct cell has no target", 0, nullptr);
    }
    if (*code != ref->target->code) {
        return reject("code cell and target disagree", 0, nullptr);
    }

    try {
        uint32_t translated_phys = ref->target->phys_addr;
        if (translate) {
            mmu_translate_imem(entry_pc, &translated_phys);
        }
        const uint32_t mode = ppc_jit_mode();
        JitBlock* cached = cache_lookup(translated_phys, mode);

        if (ref->target->mode != mode) {
            return reject("target mode is stale", translated_phys, cached);
        }
        if (translate && ref->target->phys_addr != translated_phys) {
            return reject("target physical address is stale", translated_phys, cached);
        }
        if (*phys != (translated_phys & PPC_PAGE_MASK)) {
            return reject("inline physical guard is stale", translated_phys, cached);
        }
        if (cached != ref->target) {
            return reject("cache lookup names another block", translated_phys, cached);
        }

        va_verify_successes++;
        if (va_verify_checks == 1) {
            LOG_F(INFO, "JIT VA %s verify: first direct hit validated at pc=%08X",
                  translate ? "full" : "cache-only", entry_pc);
        } else if ((va_verify_checks & ((uint64_t(1) << 24) - 1)) == 0) {
            LOG_F(INFO,
                  "JIT VA verify progress: %llu checks, %llu valid, "
                  "%llu mismatches, %llu fetch unwinds",
                  (unsigned long long)va_verify_checks,
                  (unsigned long long)va_verify_successes,
                  (unsigned long long)va_verify_failures,
                  (unsigned long long)va_verify_exceptions);
        }
        return ref->target->code;
    } catch (PPCExcUnwind&) {
        va_verify_exceptions++;
        if (va_verify_reports++ < 32) {
            LOG_F(ERROR, "JIT VA verify fetch unwound at pc=%08X way=%u",
                  entry_pc, way);
        }
        return nullptr;
    }
}

const void* rt_chain_cache_va(ChainVaSlot* slot, uint32_t way,
                              uint32_t entry_pc) noexcept {
    return validate_chain_va(slot, way, entry_pc, false);
}

const void* rt_chain_verify_va(ChainVaSlot* slot, uint32_t way,
                               uint32_t entry_pc) noexcept {
    return validate_chain_va(slot, way, entry_pc, true);
}

const void* rt_chain_call_va(const void* code, uint32_t) noexcept {
    return code;
}

const void* rt_chain_translate_va(const void* code,
                                  uint32_t entry_pc) noexcept {
    try {
        const uint32_t tag = (entry_pc & PPC_PAGE_MASK) | g_itlb_epoch;
        const TLBEntry& primary =
            pCurITLB1[(entry_pc >> PPC_PAGE_SIZE_BITS) & tlb_size_mask];
        va_translate_checks++;
        if (primary.tag == tag) {
            va_translate_primary_hits++;
        } else {
            va_translate_misses++;
        }

        if (va_translate_checks == 1) {
            LOG_F(INFO,
                  "JIT VA translate probe: first access was a primary ITLB %s",
                  primary.tag == tag ? "hit" : "miss");
        } else if ((va_translate_checks & ((uint64_t(1) << 24) - 1)) == 0) {
            LOG_F(INFO,
                  "JIT VA translate progress: %llu calls, %llu primary hits, "
                  "%llu misses",
                  (unsigned long long)va_translate_checks,
                  (unsigned long long)va_translate_primary_hits,
                  (unsigned long long)va_translate_misses);
        }

        uint32_t translated_phys = 0;
        mmu_translate_imem(entry_pc, &translated_phys);
        return code;
    } catch (PPCExcUnwind&) {
        return nullptr;
    }
}

} // namespace dppc_jit

bool ppc_jit_enable(JitBackend choice) {
    ppc_jit_disable();

    dppc_jit::va_verify_checks     = 0;
    dppc_jit::va_verify_successes  = 0;
    dppc_jit::va_verify_resolvers  = 0;
    dppc_jit::va_verify_failures   = 0;
    dppc_jit::va_verify_exceptions = 0;
    dppc_jit::va_verify_reports    = 0;
    dppc_jit::va_translate_checks       = 0;
    dppc_jit::va_translate_primary_hits = 0;
    dppc_jit::va_translate_misses       = 0;

    if (choice == JitBackend::automatic) {
        dppc_jit::native_backend = dppc_jit::make_native_backend();
        if (!dppc_jit::native_backend) {
            return false;
        }
    } else {
        // Retained as an explicit differential-test backend. It is never the
        // implicit destination of a native emitter decline.
        dppc_jit::threaded_backend = dppc_jit::make_threaded_backend();
        if (!dppc_jit::threaded_backend) {
            return false;
        }
    }

    if (getenv("DPPC_JIT_TRACE")) {
        dppc_jit::trace_enabled = true;
        // the way a wrong translation shows up is an abort deep in the MMU,
        // with nothing to say what jumped there
        loguru::set_fatal_handler([](const loguru::Message&) { dppc_jit::trace_dump(); });
        LOG_F(INFO, "JIT: tracing block entries");
    }

    if (getenv("DPPC_JIT_SYNC")) {
        dppc_jit::jit_sync_every_call = true;
        LOG_F(INFO, "JIT: settling cycles before every helper call");
    }

    if (const char* groups = getenv("DPPC_JIT_OPS")) {
        dppc_jit::jit_decode_groups =
            uint32_t(strtoul(groups, nullptr, 0)) & dppc_jit::JIT_DECODE_ALL;
        LOG_F(INFO, "JIT: decoding groups 0x%02X", dppc_jit::jit_decode_groups);
    }

    if (const char* limit = getenv("DPPC_JIT_BLOCK")) {
        const long n = strtol(limit, nullptr, 0);
        if (n >= 1 && n <= long(dppc_jit::JIT_MAX_BLOCK_INSNS)) {
            dppc_jit::jit_max_block_insns = uint32_t(n);
            LOG_F(INFO, "JIT: blocks limited to %ld guest instructions", n);
        }
    }

    if (const char* sb = getenv("DPPC_JIT_SUPERBLOCK")) {
        dppc_jit::jit_superblocks = strtol(sb, nullptr, 0) != 0;
        if (!dppc_jit::jit_superblocks) {
            LOG_F(INFO, "JIT: superblocks off, every branch ends its block");
        }
    }

    if (const char* cx = getenv("DPPC_JIT_CROSS")) {
        dppc_jit::jit_cross_follow = strtol(cx, nullptr, 0) != 0;
        if (!dppc_jit::jit_cross_follow) {
            LOG_F(INFO, "JIT: cross page walk through off");
        }
    }

    if (const char* bf = getenv("DPPC_JIT_BLRFOLLOW")) {
        dppc_jit::jit_blr_follow = strtol(bf, nullptr, 0) != 0;
        if (!dppc_jit::jit_blr_follow) {
            LOG_F(INFO, "JIT: leaf blr dissolve off");
        }
    }

    if (const char* cf = getenv("DPPC_JIT_CRFUSE")) {
        dppc_jit::jit_cr_fuse = strtol(cf, nullptr, 0) != 0;
        if (!dppc_jit::jit_cr_fuse) {
            LOG_F(INFO, "JIT: compare and branch fusion off");
        }
    }

    if (const char* chain = getenv("DPPC_JIT_CHAIN")) {
        dppc_jit::jit_chaining = strtol(chain, nullptr, 0) != 0;
        if (!dppc_jit::jit_chaining) {
            LOG_F(INFO, "JIT: direct block chaining off");
        }
    }

    if (const char* local = getenv("DPPC_JIT_CHAIN_LOCAL")) {
        dppc_jit::jit_local_chaining = strtol(local, nullptr, 0) != 0;
        if (!dppc_jit::jit_local_chaining) {
            LOG_F(INFO, "JIT: same-page block chaining off");
        }
    }

    if (const char* va = getenv("DPPC_JIT_CHAIN_VA")) {
        dppc_jit::jit_va_chaining = strtol(va, nullptr, 0) != 0;
        if (!dppc_jit::jit_va_chaining) {
            LOG_F(INFO, "JIT: virtual-address block chaining off");
        }
    }

    if (const char* revalidate = getenv("DPPC_JIT_CHAIN_VA_REVALIDATE")) {
        dppc_jit::jit_va_revalidate = strtol(revalidate, nullptr, 0) != 0;
        if (!dppc_jit::jit_va_revalidate) {
            LOG_F(INFO, "JIT: inline VA-chain ITLB revalidation off");
        }
    }

    if (const char* bind = getenv("DPPC_JIT_CHAIN_VA_BIND")) {
        dppc_jit::jit_va_binding = strtol(bind, nullptr, 0) != 0;
        if (dppc_jit::jit_va_binding) {
            LOG_F(WARNING, "JIT: experimental direct VA-chain binding enabled");
        }
    }

    dppc_jit::jit_va_publish_pc = false;
    if (const char* publish = getenv("DPPC_JIT_CHAIN_VA_PUBLISH_PC")) {
        dppc_jit::jit_va_publish_pc = strtol(publish, nullptr, 0) != 0;
        if (dppc_jit::jit_va_publish_pc) {
            LOG_F(WARNING, "JIT: publishing PC before direct VA-chain handoffs");
        }
    }

    dppc_jit::jit_va_call_probe = false;
    if (const char* call = getenv("DPPC_JIT_CHAIN_VA_CALL")) {
        dppc_jit::jit_va_call_probe = strtol(call, nullptr, 0) != 0;
        if (dppc_jit::jit_va_call_probe) {
            LOG_F(WARNING, "JIT: routing direct VA-chain handoffs through a pass-through call");
        }
    }

    dppc_jit::jit_va_translate_probe = false;
    if (const char* translate = getenv("DPPC_JIT_CHAIN_VA_TRANSLATE")) {
        dppc_jit::jit_va_translate_probe = strtol(translate, nullptr, 0) != 0;
        if (dppc_jit::jit_va_translate_probe) {
            LOG_F(WARNING, "JIT: repeating instruction translation before direct VA-chain handoffs");
        }
    }

    dppc_jit::jit_va_cache_probe = false;
    if (const char* cache = getenv("DPPC_JIT_CHAIN_VA_CACHE")) {
        dppc_jit::jit_va_cache_probe = strtol(cache, nullptr, 0) != 0;
        if (dppc_jit::jit_va_cache_probe) {
            LOG_F(WARNING,
                  "JIT: validating direct VA chains against the block cache without MMU translation");
        }
    }

    dppc_jit::jit_va_verify = false;
    if (const char* verify = getenv("DPPC_JIT_CHAIN_VA_VERIFY")) {
        dppc_jit::jit_va_verify = strtol(verify, nullptr, 0) != 0;
        if (dppc_jit::jit_va_verify) {
            LOG_F(WARNING, "JIT: authoritative direct VA-chain verification enabled");
        }
    }

    if (const char* rc = getenv("DPPC_JIT_RECYCLE")) {
        dppc_jit::jit_pool_recycle = strtol(rc, nullptr, 0) != 0;
        if (dppc_jit::jit_pool_recycle) {
            LOG_F(INFO, "JIT: dead block recycling on");
        }
    }

    if (dppc_jit::native_backend) {
        if (const char* heat = getenv("DPPC_JIT_HEAT")) {
            const long n = strtol(heat, nullptr, 0);
            if (n >= 0 && n <= 1000000) {
                dppc_jit::heat_threshold = uint32_t(n);
                if (n) {
                    LOG_F(INFO, "JIT: blocks go native after %ld entries", n);
                } else {
                    LOG_F(INFO, "JIT: heat gate off, every block goes native at once");
                }
            }
        }
    }

    dppc_jit::backend_label = dppc_jit::native_backend
        ? std::string(dppc_jit::native_backend->name()) + " over interpreter"
        : dppc_jit::threaded_backend->name();

    ppc_code_cache_set_release_cb(dppc_jit::on_block_released);
    ppc_jit_enabled = true;

    LOG_F(INFO, "JIT enabled, %s", ppc_jit_backend_name());
    return true;
}

void ppc_jit_disable() {
    ppc_jit_enabled = false;

    if ((dppc_jit::jit_va_verify || dppc_jit::jit_va_cache_probe) &&
        (dppc_jit::va_verify_checks || dppc_jit::va_verify_resolvers)) {
        LOG_F(INFO,
              "JIT VA %s verify summary: %llu direct checks, %llu valid, "
              "%llu resolver cells, %llu mismatches, %llu fetch unwinds",
              dppc_jit::jit_va_verify ? "full" : "cache-only",
              (unsigned long long)dppc_jit::va_verify_checks,
              (unsigned long long)dppc_jit::va_verify_successes,
              (unsigned long long)dppc_jit::va_verify_resolvers,
              (unsigned long long)dppc_jit::va_verify_failures,
              (unsigned long long)dppc_jit::va_verify_exceptions);
    }
    if (dppc_jit::jit_va_translate_probe &&
        dppc_jit::va_translate_checks) {
        LOG_F(INFO,
              "JIT VA translate summary: %llu calls, %llu primary ITLB hits, "
              "%llu misses",
              (unsigned long long)dppc_jit::va_translate_checks,
              (unsigned long long)dppc_jit::va_translate_primary_hits,
              (unsigned long long)dppc_jit::va_translate_misses);
    }

    dppc_jit::flush_everything();
    dppc_jit::native_backend.reset();
    dppc_jit::threaded_backend.reset();

    dppc_jit::native_compiles   = 0;
    dppc_jit::threaded_compiles = 0;
    dppc_jit::jit_va_publish_pc = false;
    dppc_jit::jit_va_call_probe = false;
    dppc_jit::jit_va_translate_probe = false;
    dppc_jit::jit_va_cache_probe = false;
    dppc_jit::jit_va_verify = false;

    ppc_code_cache_set_release_cb(nullptr);
}

void ppc_jit_flush() {
    dppc_jit::flush_everything();
}

unsigned ppc_jit_num_blocks() {
    return unsigned(dppc_jit::cache_size());
}

unsigned ppc_jit_native_compiles() {
    return dppc_jit::native_compiles;
}

unsigned ppc_jit_threaded_compiles() {
    return dppc_jit::threaded_compiles;
}

uint64_t ppc_jit_bound_chains() {
    return uint64_t(dppc_jit::chain_refs);
}

bool ppc_jit_validate_chains(std::string* error) {
    return dppc_jit::validate_chain_graph(error);
}

uint64_t ppc_jit_va_verify_checks() {
    return dppc_jit::va_verify_checks;
}

uint64_t ppc_jit_va_verify_failures() {
    return dppc_jit::va_verify_failures + dppc_jit::va_verify_exceptions;
}

const char* ppc_jit_backend_name() {
    if (!ppc_jit_enabled) {
        return nullptr;
    }
    return dppc_jit::backend_label.c_str();
}

/** Clang's function-type sanitizer expects compiler metadata before every
    indirect-call target. The sole indirect call below enters generated code,
    which cannot carry that metadata. Other ASan/UBSan checks stay enabled. */
#if defined(__clang__)
__attribute__((no_sanitize("function")))
#endif
void ppc_jit_exec_inner(JitExecType type, uint32_t goal_addr) {
    exec_flags = 0;

    // rt_dispatch applies the same stopping rule from inside the frame the
    // blocks share, so it has to know what this run is
    dppc_jit::run_type = type;
    dppc_jit::run_goal = goal_addr;

    // an `until` run observes the PC at every block entry, and a chain bound
    // in an earlier run would let blocks hand off without being seen, so the
    // bindings go back to their resolvers. They rebind lazily on the next
    // ordinary run. Tracing wants every entry for the same reason
    dppc_jit::chain_allowed = (type == JitExecType::run) && !dppc_jit::trace_enabled;
    if (!dppc_jit::chain_allowed) {
        dppc_jit::unbind_all_chains();
    }

    while (power_on) {
        // no block is executing here, so anything invalidation unlinked while
        // the last one ran can go back now
        dppc_jit::drain_pending_free();

        const uint32_t entry_pc = ppc_state.pc;
        const uint8_t* interp_code = nullptr;
        bool cold_fallback = false;
        dppc_jit::JitBlock* blk =
            dppc_jit::find_or_translate(
                entry_pc, true, true, &interp_code, &cold_fallback);

        dppc_jit::trace_block(blk);

        if (blk && dppc_jit::goal_splits_block(blk, entry_pc)) {
            blk = nullptr;
        }

        if (blk) {
            // a native block does not come back after one block: it enters
            // its own frame and keeps going through rt_dispatch until
            // something here has to be dealt with
            blk->entry(blk);
        } else {
            if (cold_fallback && !dppc_jit::trace_enabled) {
                dppc_jit::interpret_cold_span(interp_code);
            } else {
                dppc_jit::interpret_one(interp_code);
            }
        }

        if (exec_flags) {
            // same handling ppc_exec_inner does, minus the page bookkeeping:
            // the next lookup re-reads the PC anyway, and a decoder change is
            // already covered because the opcode table is part of the block key
            if ((exec_flags & EXEF_SLEEP) && !(exec_flags & EXEF_EXCEPTION)) [[unlikely]] {
                while (power_on && (exec_flags & EXEF_SLEEP)) {
                    g_icycles_max = ppc_process_events();
                    if (!(exec_flags & EXEF_SLEEP)) {
                        break;
                    }
                    if (g_icycles_max > g_icycles) {
                        g_icycles = g_icycles_max;
                    } else {
                        g_icycles++;
                    }
                }
            }
            ppc_state.pc = ppc_next_instruction_address;
            exec_flags   = 0;
        }

        if (type == JitExecType::until) {
            if (ppc_state.pc == goal_addr) {
                break;
            }
        }
    }
}
