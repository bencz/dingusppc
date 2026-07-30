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

/** Both names, because both are in play: an emitter that declines a block
    does not send it to the interpreter, it sends it to the threaded backend */
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

/** Entries a block collects as a threaded one before the emitter is asked.

    A booting system runs megabytes of code exactly once: loaders, linkers,
    initialisation that never comes back. Emitting all of it cost more than
    running it, and the churn was worse than the cost: every block thrown
    away by an invalidation had bought chain bindings, registry entries and
    code pool bytes with it. Below the threshold a block runs as IR, which
    is roughly interpreter speed; crossing it buys emission once.

    Zero turns the gate off and every block compiles native on first entry,
    which is what the tests use to hold the emitter to full coverage.
    DPPC_JIT_HEAT overrides it for bisection either way */
uint32_t promote_threshold = 8;

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
void unlink_chain_ref(ChainRef* ref) {
    JitBlock* blk = ref->target;
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
    chain_refs--;
}

/** Points a slot's registry entry at a new target. A rebinding entry moves
    off the old target's list first, so nothing is allocated and nothing is
    left behind, however hard a site churns */
void bind_chain(JitBlock* blk, ChainRef* ref, void** code, const void* resolver,
                uint64_t* pred) {
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
    chain_refs++;
}

/** Sends every slot aimed at the block back to resolving */
void unbind_chains_to(JitBlock* blk) {
    ChainRef* ref = blk->chain_in;
    if (!ref) {
        return;
    }
    do {
        *ref->code = const_cast<void*>(ref->resolver);
        if (ref->pred) {
            *ref->pred = 1;
        }
        ref->target = nullptr;
        chain_refs--;
        ref = ref->next;
    } while (ref);
    blk->chain_in = nullptr;
    unlink_chained_block(blk);
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
    if this is the first time. Returns nullptr when the instruction there has
    to go back to the interpreter.

    allow_flush says the caller sits below the code memory floor, which the
    dispatch stub does and a resolver thunk does not: a thunk lives in the
    block that emitted it, a flush reclaims that memory, and the very next
    compile writes new code over the return address the resolver call left
    on the stack. A resolver therefore passes false and leaves the block
    untranslated rather than flushing; its exit falls back to the stub, and
    the stub's own lookup does the flushing on safe ground.

    Can unwind out of mmu_translate_imem on an instruction fetch fault, which
    is fine: no generated frame is on the stack at this point */
/** Retranslates a heated threaded block and swaps a native one into its
    place: same cache key, same invalidation registration, new executor.
    Returns nullptr when the emitter declines or the pool is full in a
    context that may not flush, and the threaded block stays as it was */
JitBlock* promote_block(JitBlock* blk, uint32_t virt_addr, uint32_t phys_addr,
                        const uint8_t* code, uint32_t mode, bool allow_flush) {
    if (!translate_block(virt_addr, phys_addr, code, mode, scratch_ir)) {
        return nullptr;
    }

    JitBlock* nblk = native_backend->compile(scratch_ir);

    if (!nblk && native_backend->wants_flush()) {
        if (!allow_flush) {
            return nullptr; // a resolver is on the stack, see find_or_translate
        }
        LOG_F(INFO, "JIT: code memory exhausted, flushing %zu blocks", cache_size());
        flush_everything(); // takes the threaded original with it
        nblk = native_backend->compile(scratch_ir);
        if (!nblk) {
            return nullptr;
        }
        native_compiles++;
        nblk->owner = native_backend.get();
        cache_insert(nblk);
        register_ranges(nblk);
        return nblk;
    }
    if (!nblk) {
        return nullptr; // declined for real; the block keeps running as IR
    }

    native_compiles++;
    nblk->owner = native_backend.get();

    cache_forget(blk);
    cache_insert(nblk);
    // not a swap in place: the retranslation may cover different ground
    // than the threaded original did, a cross page walk through forming
    // once the target's translation warmed up being the usual way, so the
    // old ranges go and the new ones are registered from scratch
    ppc_code_cache_remove(blk->phys_addr, static_cast<CodeBlockHandle>(blk));
    if (blk->second_size) {
        ppc_code_cache_remove(blk->second_phys, static_cast<CodeBlockHandle>(blk));
    }
    register_ranges(nblk);
    blk->owner->release(blk); // a threaded payload is plain heap, and the
                              // block is not executing at any promotion site
    return nblk;
}

JitBlock* find_or_translate(uint32_t virt_addr, bool allow_flush) {
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

    if (JitBlock* blk = cache_lookup(phys_addr, mode)) {
        // a threaded block heats up with every entry; the one that crosses
        // the threshold comes back native, and the caller jumps straight
        // into it. A failed promotion leaves the threaded block in place,
        // and resetting the heat spaces out the retries
        if (blk->owner == threaded_backend.get() && native_backend &&
            promote_threshold && ++blk->heat >= promote_threshold) {
            blk->heat = 0;
            if (JitBlock* promoted =
                    promote_block(blk, virt_addr, phys_addr, code, mode, allow_flush)) {
                return promoted;
            }
        }
        return blk;
    }

    if (!translate_block(virt_addr, phys_addr, code, mode, scratch_ir)) {
        return nullptr;
    }

    // with a native backend present a block is born threaded and earns its
    // emission through the heat gate above; without one, threaded is all
    // there is. The gate off means native on first entry
    Backend* owner = native_backend.get();
    JitBlock* blk  = nullptr;

    if (owner && !promote_threshold) {
        blk = owner->compile(scratch_ir);

        if (!blk && owner->wants_flush()) {
            if (!allow_flush) {
                // a resolver is on the stack with its return address inside
                // the memory a flush reclaims. No block, no threaded
                // fallback, no cache entry: the retry from the dispatch
                // stub redoes all of it
                return nullptr;
            }
            // the pool filled up. Everything translated so far goes, and
            // this block gets one more try on the empty pool. Nothing is
            // executing here, so the code the flush reclaims cannot be
            // under anyone's feet
            LOG_F(INFO, "JIT: code memory exhausted, flushing %zu blocks", cache_size());
            flush_everything();
            blk = owner->compile(scratch_ir);
        }
    }

    if (blk) {
        native_compiles++;
    } else {
        owner = threaded_backend.get();
        blk   = owner->compile(scratch_ir);
        if (!blk) {
            return nullptr;
        }
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
    Leaves the PC and exec_flags exactly as a block would */
void interpret_one() {
    uint8_t* pc_real = mmu_translate_imem(ppc_state.pc);
    uint32_t opcode  = ppc_read_instruction(pc_real);

    ppc_main_opcode(ppc_opcode_grabber, opcode);
    ppc_account_cycles(1);

    if (!exec_flags) {
        ppc_state.pc += 4;
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
    JitBlock* blk = find_or_translate(entry_pc, true);
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
        JitBlock* blk = find_or_translate(entry_pc, false);
        trace_block(blk);

        if (!blk || !blk->code || goal_splits_block(blk, entry_pc)) [[unlikely]] {
            return nullptr;
        }

        // the jump itself is valid either way; what `until` and tracing veto
        // is the binding, which would let later passes skip this observation
        if (chain_allowed) {
            // the cell still points at the resolver thunk, or this call
            // would not be happening
            bind_chain(blk, &slot->ref, &slot->code, slot->code, nullptr);
            slot->code = blk->code;
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
        JitBlock* blk = find_or_translate(entry_pc, false);
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
        if (chain_allowed) {
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
        }
        return blk->code;
    } catch (PPCExcUnwind&) {
        return nullptr;
    }
}

} // namespace dppc_jit

bool ppc_jit_enable(JitBackend choice) {
    ppc_jit_disable();

    if (choice == JitBackend::automatic) {
        dppc_jit::native_backend = dppc_jit::make_native_backend();
    }

    // present either way: it is what takes the blocks the emitter declines
    dppc_jit::threaded_backend = dppc_jit::make_threaded_backend();
    if (!dppc_jit::threaded_backend) {
        return false;
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

    if (const char* heat = getenv("DPPC_JIT_HEAT")) {
        const long n = strtol(heat, nullptr, 0);
        if (n >= 0 && n <= 1000000) {
            dppc_jit::promote_threshold = uint32_t(n);
            if (n) {
                LOG_F(INFO, "JIT: blocks go native after %ld entries", n);
            } else {
                LOG_F(INFO, "JIT: heat gate off, every block goes native at once");
            }
        }
    }

    dppc_jit::backend_label = dppc_jit::native_backend
        ? std::string(dppc_jit::native_backend->name()) + " over " +
          dppc_jit::threaded_backend->name()
        : dppc_jit::threaded_backend->name();

    ppc_code_cache_set_release_cb(dppc_jit::on_block_released);
    ppc_jit_enabled = true;

    LOG_F(INFO, "JIT enabled, %s", ppc_jit_backend_name());
    return true;
}

void ppc_jit_disable() {
    ppc_jit_enabled = false;

    dppc_jit::flush_everything();
    dppc_jit::native_backend.reset();
    dppc_jit::threaded_backend.reset();

    dppc_jit::native_compiles   = 0;
    dppc_jit::threaded_compiles = 0;

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

const char* ppc_jit_backend_name() {
    if (!ppc_jit_enabled) {
        return nullptr;
    }
    return dppc_jit::backend_label.c_str();
}

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
        dppc_jit::JitBlock* blk = dppc_jit::find_or_translate(entry_pc, true);

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
            dppc_jit::interpret_one();
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
