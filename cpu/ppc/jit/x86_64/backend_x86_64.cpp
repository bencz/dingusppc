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

/** @file x86-64 emitter.

    Registers pinned for as long as generated code is running, all callee
    saved so the helpers cannot lose them:

      rbx  &ppc_state
      r12  the PC the current block was entered through
      r13  &exec_flags
      r14  &rt_call_op
      r15  guest instructions the current block has retired

    Everything else volatile is a pool for IR values, minus r11, which stays
    free as the scratch a call through an absolute address needs.

    A block is not a function. Two shared stubs are emitted once, ahead of
    every block, and the frame belongs to them:

      the trampoline is what JitBlock::entry points at. It is called from C++,
      saves the five pinned registers, fills them in and jumps to the block

      the dispatch stub is where every block exit jumps. It asks rt_dispatch
      where to go next and either jumps there, still on the same frame, or
      unwinds the frame and returns to C++

    That is the whole reason for the split. With blocks averaging 4.3 guest
    instructions, a prologue and an epilogue per block cost more than the
    block: five pushes, three ten byte immediates, a call to close the block
    out, five pops and a return, against roughly four real instructions. Paid
    once per entry into generated code instead, they disappear.

    What a block therefore owes on the way out: ppc_state.pc pointing at the
    guest instruction that did not run, r15 holding how many did, and a jump
    to the dispatch stub. Nothing else.

    Allocation is greedy over the block and needs no spill logic, which rests
    on a property the translator guarantees: no IR value is ever live across a
    Call. A Call is always preceded by the writeback of every cached guest
    register, which consumes the values, and followed by dropping the cache,
    so nothing defined before it can be read after. If the pool ever runs dry
    the block is declined and the interpreter takes it, which is the same
    escape hatch used for opcodes outside the emitted subset.
 */

#include "../abi.h"
#include "../backend.h"
#include "../jitruntime.h"
#include "../host/codemem.h"
#include "../../ppcmmu.h"
#include "emitter.h"

#include <loguru.hpp>

namespace {

/** DPPC_JIT_MAPLOG=<path> writes one line per emitted block: host address,
    host bytes, guest instructions, end reason, entry address and the raw
    word that closed the block. Joining it against perf samples is what
    turns a flat [JIT] profile into a histogram of block shapes, which is
    the evidence superblock decisions run on. FLUSH marks a pool reset,
    after which host addresses start meaning different blocks */
FILE* map_log() {
    static FILE* f = [] {
        const char* path = getenv("DPPC_JIT_MAPLOG");
        return path ? fopen(path, "w") : nullptr;
    }();
    return f;
}

} // namespace

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dppc_jit {

namespace {

constexpr uint32_t bit(uint8_t reg) {
    return 1u << reg;
}

const uint8_t sysv_args[] = {RDI, RSI, RDX, RCX, R8, R9};
const uint8_t win64_args[] = {RCX, RDX, R8, R9};

const AbiDesc abi_sysv = {
    /* name              */ "x86-64 System V",
    /* volatile_gprs     */ bit(RAX) | bit(RCX) | bit(RDX) | bit(RSI) | bit(RDI) |
                            bit(R8)  | bit(R9)  | bit(R10) | bit(R11),
    /* callee_saved_gprs */ bit(RBX) | bit(R12) | bit(R13) | bit(R14) | bit(R15),
    /* reserved_gprs     */ bit(RSP) | bit(RBP),
    /* int_arg_regs      */ sysv_args,
    /* num_int_arg_regs  */ 6,
    /* int_ret_reg       */ RAX,
    /* stack_alignment   */ 16,
    /* shadow_space      */ 0,
};

const AbiDesc abi_win64 = {
    /* name              */ "x86-64 Microsoft",
    /* volatile_gprs     */ bit(RAX) | bit(RCX) | bit(RDX) |
                            bit(R8)  | bit(R9)  | bit(R10) | bit(R11),
    /* callee_saved_gprs */ bit(RBX) | bit(RSI) | bit(RDI) |
                            bit(R12) | bit(R13) | bit(R14) | bit(R15),
    /* reserved_gprs     */ bit(RSP) | bit(RBP),
    /* int_arg_regs      */ win64_args,
    /* num_int_arg_regs  */ 4,
    /* int_ret_reg       */ RAX,
    /* stack_alignment   */ 16,
    /* shadow_space      */ 32,
};

/** Saved and restored around every block, in this order */
const X64Gpr pinned[] = {RBX, R12, R13, R14, R15};
constexpr size_t NUM_PINNED = sizeof(pinned) / sizeof(pinned[0]);

constexpr X64Gpr REG_STATE   = RBX;
constexpr X64Gpr REG_ENTRYPC = R12;
constexpr X64Gpr REG_FLAGS   = R13;
constexpr X64Gpr REG_CALLOP  = R14;
constexpr X64Gpr REG_RETIRED = R15;
constexpr X64Gpr REG_SCRATCH = R11;

/** REG_RETIRED is zero at every generated block entry. The trampoline and
    dispatch stub establish the invariant; a block only writes a nonzero value
    on a terminal path straight to dispatch, so chained exits may preserve it
    instead of zeroing an already-zero callee-saved register. */

/** The timing globals are reached as displacements from &ppc_state rather
    than each through an immediate of its own, which would be ten bytes and a
    scratch register apiece on a path taken before every helper call.

    They are ordinary statics in the same image, so the distance is kilobytes
    and a displacement covers it with room to spare. init measures rather than
    assumes, and declines the backend if it ever turns out otherwise */
constexpr X64Gpr REG_TIME = REG_STATE;

inline int64_t disp_from_state(const void* p) {
    return int64_t(uintptr_t(p)) - int64_t(uintptr_t(&ppc_state));
}

constexpr int32_t PC_OFFSET  = int32_t(offsetof(SetPRS, pc));
constexpr int32_t CR_OFFSET  = int32_t(offsetof(SetPRS, cr));
constexpr int32_t CTR_OFFSET = int32_t(offsetof(SetPRS, spr) + SPR::CTR * sizeof(uint32_t));
constexpr int32_t LR_OFFSET  = int32_t(offsetof(SetPRS, spr) + SPR::LR  * sizeof(uint32_t));
constexpr int32_t XER_OFFSET = int32_t(offsetof(SetPRS, spr) + SPR::XER * sizeof(uint32_t));

inline int32_t gpr_offset(unsigned reg) {
    return int32_t(offsetof(SetPRS, gpr) + reg * sizeof(uint32_t));
}

inline int32_t spr_offset(unsigned spr) {
    return int32_t(offsetof(SetPRS, spr) + spr * sizeof(uint32_t));
}

int32_t frame_padding(const AbiDesc& abi) {
    const size_t after_pushes = 8 + NUM_PINNED * 8;
    const size_t misaligned   = after_pushes % abi.stack_alignment;
    const int32_t pad = misaligned ? int32_t(abi.stack_alignment - misaligned) : 0;
    return pad + int32_t(abi.shadow_space);
}

/** Fields of the primary data TLB the fast path reads. Confirmed against the
    struct rather than assumed: 32 bytes per entry, tag first, the host
    address offset eight in */
constexpr int32_t TLB_TAG_OFFSET  = int32_t(offsetof(TLBEntry, tag));
constexpr int32_t TLB_HOSTR_OFFSET = int32_t(offsetof(TLBEntry, host_va_offs_r));
constexpr int32_t TLB_HOSTW_OFFSET = int32_t(offsetof(TLBEntry, host_va_offs_w));
constexpr int32_t TLB_FLAGS_OFFSET = int32_t(offsetof(TLBEntry, flags));
constexpr int32_t TLB_PHYS_OFFSET  = int32_t(offsetof(TLBEntry, phys_tag));

/** What the write fast path insists on. Without PAGE_WRITABLE the page is
    either read only, which is a DSI, or it holds translated code and the
    blocks have to go first. Without PTE_SET_C the change bit of the PTE still
    needs updating. Both are work the emitted path does not do */
constexpr uint32_t TLB_STORE_READY = TLBFlags::PAGE_WRITABLE | TLBFlags::PTE_SET_C;
constexpr uint8_t TLB_ENTRY_SHIFT = 5; // sizeof(TLBEntry) == 32

constexpr size_t MAX_BLOCK_BYTES = 64 * 1024;
constexpr uint8_t NO_REG = 0xFF;

class X86_64Backend : public Backend {
public:
    explicit X86_64Backend(const AbiDesc& abi)
        : abi(abi), pad(frame_padding(abi)),
          pool(abi_usable_gprs(abi) & abi.volatile_gprs & ~bit(REG_SCRATCH)) {}

    bool init() {
        if (!this->measure_time_globals()) {
            LOG_F(WARNING, "JIT: timing globals too far from the register file");
            return false;
        }
        // the tail holds the chain slots, a few dozen bytes per block worst
        // case. The pool is a reservation, not a footprint: physical pages
        // arrive as emission touches them and go back at every flush, so
        // generosity here costs address space only. What it buys is flush
        // scarcity, because nothing is reclaimed per block and gross
        // emission is what fills it; the Cheetah storm emits past 128 MB.
        // DPPC_JIT_POOL shrinks it deliberately, which is how the flush
        // paths get exercised on demand instead of once in a blue moon
        size_t pool_mb = 512;
        if (const char* env = getenv("DPPC_JIT_POOL")) {
            const long n = strtol(env, nullptr, 0);
            if (n >= 1 && n <= 4096) {
                pool_mb = size_t(n);
                LOG_F(INFO, "JIT: code pool limited to %zu MB", pool_mb);
            }
        }
        if (!this->code.init(pool_mb * 1024 * 1024, (pool_mb * 1024 * 1024) / 4)) {
            return false;
        }
        return this->emit_shared_stubs();
    }

    const char* name() const override {
        return this->abi.name;
    }

    JitBlock* compile(const IRBlock& ir) override {
        if (!this->emit(ir)) {
            return nullptr;
        }

        uint8_t* dst = this->code.alloc_writable(this->asmb.size());
        if (!dst) {
            // The region is full. The flag asks the facade for a whole-pool
            // flush before it gives this block back to the interpreter.
            this->full = true;
            LOG_F(INFO, "JIT: code memory full after %zu bytes", this->code.used());
            return nullptr;
        }

        std::memcpy(dst, this->asmb.bytes(), this->asmb.size());

        // the jumps to the dispatch stub are the one thing that could not be
        // encoded while the bytes were floating, since rel32 is measured from
        // where the instruction ends up
        const bool relocated = this->asmb.relocate(dst);

        if (!this->code.end_write_range()) {
            return nullptr;
        }
        if (!relocated) {
            LOG_F(WARNING, "JIT: dispatch stub out of rel32 range, block declined");
            return nullptr;
        }

        // now that the code has a home, every chain slot can point at its
        // resolver thunk. The slots live in plain data, no protection dance
        for (const ChainExit& ce : this->chain_exits) {
            ce.slot->code = dst + ce.thunk_off;
        }
        for (const VaChainExit& ce : this->va_chain_exits) {
            ce.slot->code0    = dst + ce.thunk_off;
            ce.slot->code1    = dst + ce.thunk_off;
            ce.slot->resolver = dst + ce.thunk_off;
        }

        JitBlock* blk   = new JitBlock;
        blk->virt_addr  = ir.virt_addr;
        blk->phys_addr  = ir.phys_addr;
        blk->mode       = ir.mode;
        blk->byte_size  = ir.byte_size;
        blk->end_off    = ir.end_off;
        blk->second_phys = ir.second_phys;
        blk->second_size = ir.second_size;
        blk->insn_count = ir.insn_count;
        blk->end_reason = uint8_t(ir.end_reason);
        blk->entry      = this->trampoline;
        blk->payload    = nullptr; // the code is the payload, and it is pooled
        blk->code       = dst;
        blk->code_bytes = uint32_t(this->asmb.size());
        blk->owner      = nullptr; // the facade fills this in
        blk->chain_in     = nullptr;
        blk->chain_out    = nullptr;
        blk->chained_prev = nullptr;
        blk->chained_next = nullptr;

        // The entries live in this block's slot area but sit on their
        // targets' incoming lists while bound. Keep an ownership list so
        // invalidating the source can detach them before that memory dies.
        for (const ChainExit& ce : this->chain_exits) {
            ce.slot->ref.owner_next = blk->chain_out;
            blk->chain_out          = &ce.slot->ref;
        }
        for (const VaChainExit& ce : this->va_chain_exits) {
            ce.slot->ref0.owner_next = blk->chain_out;
            blk->chain_out           = &ce.slot->ref0;
            ce.slot->ref1.owner_next = blk->chain_out;
            blk->chain_out           = &ce.slot->ref1;
        }

        if (FILE* f = map_log()) [[unlikely]] {
            fprintf(f, "%llx %zx %u %u %08x %08x %x\n",
                    (unsigned long long)(uintptr_t)dst, this->asmb.size(),
                    ir.insn_count, unsigned(ir.end_reason), ir.virt_addr,
                    ir.end_word, ir.byte_size);
        }

        return blk;
    }

    void release(JitBlock* blk) override {
        // the dead block's bytes feed the next compile of their size. The
        // caller only drains at a block boundary, where nothing is executing
        // them and every chain into the block was already unbound. Off under
        // the map log, whose addresses must keep meaning one block each
        if (jit_pool_recycle && blk->code_bytes && !map_log()) {
            this->code.recycle(static_cast<uint8_t*>(blk->code), blk->code_bytes);
        }
        delete blk;
    }

    void release_all() override {
        this->code.reset();
        this->full = false;
        if (FILE* f = map_log()) [[unlikely]] {
            fprintf(f, "FLUSH\n");
        }
    }

    bool wants_flush() const override {
        return this->full;
    }

private:
    bool emit(const IRBlock& ir) {
        // no IR at all is still a block: eieio decodes to nothing and only
        // counts as retired, so the exit below is the whole body
        this->asmb.clear();
        this->cold_exits.clear();
        this->sync_exits.clear();
        this->chain_exits.clear();
        this->va_chain_exits.clear();
        this->accounted = 0;
        this->cur_virt_addr = ir.virt_addr;
        this->live_cmp.valid = false;
        this->plan_lifetimes(ir);

        const X64Gpr arg0 = X64Gpr(this->abi.int_arg_regs[0]);
        const X64Gpr arg1 = X64Gpr(this->abi.int_arg_regs[1]);
        const X64Gpr arg2 = X64Gpr(this->abi.int_arg_regs[2]);
        const X64Gpr ret  = X64Gpr(this->abi.int_ret_reg);

        uint32_t free_mask = this->pool;

        size_t dead_store_count = 0;
        for (uint8_t dead : this->dead_gpr_store) {
            dead_store_count += dead != 0;
        }
        // Removing a long run of state stores can move the body and
        // its chained exit into a bad 32-byte frontend phase. It showed up as
        // a repeatable 9-16% loss in dense same-page and alternating-VA loops
        // despite retiring fewer uops. A small prefix disperses those shapes
        // without giving the stores back. Dynamic work still falls at the
        // threshold: eight store uops become four NOPs, and longer runs need
        // only half as much padding.
        if (dead_store_count >= 8) {
            this->asmb.nop8();
            this->asmb.nop8();
            if (dead_store_count < 24) {
                this->asmb.nop8();
                this->asmb.nop8();
            }
        }

        for (size_t i = 0; i < ir.insns.size(); i++) {
            const IRInsn& in = ir.insns[i];

            // whether the FLAGS a fusible compare left behind survive this
            // instruction: the pure moves do, everything else scratches
            // them. SetCR manages its own record and Branch consumes it,
            // clobbering only on the CTR path it emits itself
            if (in.opcode != IROpcode::SetCR && in.opcode != IROpcode::Branch &&
                !flags_survive(in.opcode)) {
                this->live_cmp.valid = false;
            }

            if (in.opcode == IROpcode::SetCR) {
                if (!this->emit_set_cr(in, i, free_mask)) {
                    return false;
                }
                continue;
            }

            if (in.opcode == IROpcode::Store) {
                if (!this->emit_store(in, i, free_mask, arg0, arg1, arg2)) {
                    return false;
                }
                continue;
            }

            if (in.opcode == IROpcode::Load) {
                if (!this->emit_load(in, i, free_mask, arg0, arg1, ret)) {
                    return false;
                }
                continue;
            }

            if (in.opcode == IROpcode::Branch) {
                if (in.flags & IR_BRANCH_SIDE_EXIT) {
                    // taken leaves through the chain with this instruction
                    // retired, not taken falls through to the rest of the
                    // block. The builder invalidated its cache, so nothing
                    // lives across this and the exit scratches freely
                    this->emit_branch(in, in.insn_idx + 1 - this->accounted);
                    free_mask = this->pool;
                    continue;
                }
                this->emit_branch(in, ir.insn_count - this->accounted);
                // it ends the block, so what follows is only the tail below
                continue;
            }

            if (in.opcode == IROpcode::Call) {
                if (!this->emit_call(in, arg0, arg1)) {
                    return false;
                }
                // nothing is live here, so the pool comes back whole
                free_mask = this->pool;
                continue;
            }

            if (in.opcode == IROpcode::ItransGuard) {
                this->emit_itrans_guard(in);
                // the probe scratches RAX, RDX and the scratch register;
                // the builder invalidated its cache to match
                free_mask = this->pool;
                continue;
            }

            if (in.opcode == IROpcode::AddCA || in.opcode == IROpcode::AddECA ||
                in.opcode == IROpcode::SubCA || in.opcode == IROpcode::SubECA) {
                if (!this->emit_carry(in, i, free_mask)) {
                    return false;
                }
                continue;
            }

            if (in.opcode == IROpcode::MtCrf) {
                if (!this->emit_mtcrf(in, i, free_mask)) {
                    return false;
                }
                continue;
            }

            // the OE forms of the plain arithmetic; the carry family above
            // deals with its own OE inside emit_carry
            if (in.oe) {
                if (!this->emit_ov_op(in, i, free_mask)) {
                    return false;
                }
                continue;
            }

            if (!this->emit_value_op(in, i, free_mask)) {
                return false;
            }
        }

        // fell off the end, so every guest instruction retired and whatever
        // sits just past the block is next. Chainable when it cannot have
        // changed the translation mode: a not taken branch, an arbitrary
        // length cut, or the edge of the page, whose fall through takes the
        // guarded kind since it leaves the page by definition. A context
        // sync might resume under another mode and an untranslatable next
        // instruction would make every resolve fail
        const uint32_t tail_retired = ir.insn_count - this->accounted;
        const bool mode_safe = ir.end_reason == BlockEnd::Branch ||
                               ir.end_reason == BlockEnd::SizeLimit ||
                               ir.end_reason == BlockEnd::PageEnd;
        bool tail_done = false;
        if (ir.end_reason == BlockEnd::Branch || ir.end_reason == BlockEnd::SizeLimit) {
            tail_done = this->emit_chained_exit(ir.end_off, tail_retired);
        }
        if (!tail_done && mode_safe) {
            this->asmb.lea_reg_mem(RDX, REG_ENTRYPC, ir.end_off);
            tail_done = this->emit_va_chained_exit(tail_retired);
        }
        if (!tail_done) {
            this->emit_exit(ir.end_off, tail_retired);
        }

        this->emit_cold_exits(ret);
        this->emit_sync_exits();
        this->emit_chain_thunks();

        if (!this->asmb.finalize()) {
            ABORT_F("JIT: a jump was emitted to a label that was never bound");
        }

        return this->asmb.size() <= MAX_BLOCK_BYTES;
    }

    /** Leaves the block for the dispatch stub, with the next guest PC at
        `next_off` bytes from the address the block was entered through.

        Nothing is live at any exit: guest register writes go straight to
        memory, so rax is free scratch here */
    void emit_exit(int32_t next_off, uint32_t retired) {
        this->asmb.lea_reg_mem(RAX, REG_ENTRYPC, next_off);
        this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, RAX);
        this->asmb.mov_reg_imm32(REG_RETIRED, retired);
        this->asmb.jmp_abs(this->dispatch);
    }

    /** Leaves the block straight into whatever the chain slot points at,
        skipping dispatch, lookup and instruction address translation.

        Only for a next PC inside the block's own guest page: the offset
        within a page is the same under every mapping of it, so a target
        proven same page at translation time stays right however the page is
        reached later, and a mode change ends the block before its exits can
        run again. That is the whole correctness argument, and it is why the
        slot starts out pointing at a resolver thunk instead of being filled
        eagerly: rt_chain_resolve redoes the full lookup once and binds only
        what it verified.

        What dispatch did for the block still has to happen here: cycles are
        settled inline and a due timer bails to the stub, with the count
        zeroed since it was already handed over. The entry PC register moves
        forward, which is all the next block needs; ppc_state.pc stays where
        it was because everything that can observe it writes it first.

        Returns false when the slot area is full, and the caller falls back
        to a plain exit */
    bool emit_chained_exit(int32_t next_off, uint32_t retired) {
        const uint32_t page_off = this->cur_virt_addr & ~PPC_PAGE_MASK;
        if ((page_off + uint32_t(next_off)) >= PPC_PAGE_SIZE) {
            return false; // leaves the page, dispatch has to translate
        }

        ChainSlot* slot =
            reinterpret_cast<ChainSlot*>(this->code.slot_alloc(sizeof(ChainSlot), 8));
        if (!slot) {
            return false;
        }
        slot->ref.target = nullptr; // unbound until the resolver says so

        X64Emitter::Label to_dispatch = this->asmb.new_label();

        if (retired) {
            this->asmb.add_mem64_imm32(REG_TIME, this->icycles_disp, retired);
        }
        this->asmb.mov_reg64_mem(RAX, REG_TIME, this->icycles_disp);
        this->asmb.cmp_reg64_mem(RAX, REG_TIME, this->deadline_disp);
        this->asmb.jcc(X64Cond::Above, to_dispatch);
        this->asmb.cmp_mem8_imm8(REG_TIME, this->timer_disp, 0);
        this->asmb.jcc(X64Cond::NotEqual, to_dispatch);

        this->asmb.lea_reg_mem(REG_ENTRYPC, REG_ENTRYPC, next_off);
        this->asmb.jmp_mem_abs(&slot->code);

        // a due timer goes the long way round; the count is zero because the
        // cycles are already in, and rt_dispatch settling zero is a no-op
        // that still lets the timers run
        this->asmb.bind(to_dispatch);
        this->asmb.lea_reg_mem(RAX, REG_ENTRYPC, next_off);
        this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, RAX);
        this->asmb.xor_reg_reg32(REG_RETIRED, REG_RETIRED);
        this->asmb.jmp_abs(this->dispatch);

        this->chain_exits.push_back({slot, 0});
        return true;
    }

    /** Leaves the block for a target that is only a virtual address: the
        registers behind blr and bcctr, an absolute branch, or a direct one
        into another guest page. RDX must hold the target address, already
        masked when it came from LR or CTR.

        Two guards stand before the bound jump: the address itself against
        the prediction, and the translation generation against the one the
        binding was made under. The generation is what makes a virtual keyed
        binding safe at all: any tlbie, BAT or SDR1 write moves it, and the
        stale binding fails here instead of running whatever code the old
        mapping had there. Everything else mirrors the same page exit */
    bool emit_va_chained_exit(uint32_t retired) {
        ChainVaSlot* slot =
            reinterpret_cast<ChainVaSlot*>(this->code.slot_alloc(sizeof(ChainVaSlot), 8));
        if (!slot) {
            return false;
        }
        slot->pred0 = 1; // matches nothing until the resolver binds
        slot->gen0  = 0;
        slot->phys0 = 0;
        slot->pred1 = 1;
        slot->gen1  = 0;
        slot->phys1 = 0;
        slot->flip  = 0;
        slot->ref0.target = nullptr; // both ways unbound
        slot->ref1.target = nullptr;

        X64Emitter::Label to_dispatch = this->asmb.new_label();
        X64Emitter::Label try_way1    = this->asmb.new_label();
        X64Emitter::Label probe0      = this->asmb.new_label();
        X64Emitter::Label probe1      = this->asmb.new_label();
        X64Emitter::Label thunk       = this->asmb.new_label();

        if (retired) {
            this->asmb.add_mem64_imm32(REG_TIME, this->icycles_disp, retired);
        }
        this->asmb.mov_reg64_mem(RAX, REG_TIME, this->icycles_disp);
        this->asmb.cmp_reg64_mem(RAX, REG_TIME, this->deadline_disp);
        this->asmb.jcc(X64Cond::Above, to_dispatch);
        this->asmb.cmp_mem8_imm8(REG_TIME, this->timer_disp, 0);
        this->asmb.jcc(X64Cond::NotEqual, to_dispatch);

        // the entry PC register moves first so the thunk and the bound jumps
        // agree on it; the low half of each prediction is the compare, the
        // layout being little endian. The resolver keeps the predictions
        // distinct, so a matching way with a stale generation means no other
        // way can match and the probe is the right place to go
        this->asmb.mov_reg_reg32(REG_ENTRYPC, RDX);
        this->asmb.cmp_reg_mem32_abs(RDX, &slot->pred0);
        this->asmb.jcc(X64Cond::NotEqual, try_way1);
        this->asmb.mov_reg64_mem_abs(RAX, &slot->gen0);
        this->asmb.cmp_reg64_mem(RAX, REG_TIME, this->gen_disp);
        this->asmb.jcc(X64Cond::NotEqual, probe0);
        this->asmb.jmp_mem_abs(&slot->code0);

        this->asmb.bind(try_way1);
        this->asmb.cmp_reg_mem32_abs(RDX, &slot->pred1);
        this->asmb.jcc(X64Cond::NotEqual, thunk);
        this->asmb.mov_reg64_mem_abs(RAX, &slot->gen1);
        this->asmb.cmp_reg64_mem(RAX, REG_TIME, this->gen_disp);
        this->asmb.jcc(X64Cond::NotEqual, probe1);
        this->asmb.jmp_mem_abs(&slot->code1);

        this->asmb.bind(to_dispatch);
        this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, RDX);
        this->asmb.xor_reg_reg32(REG_RETIRED, REG_RETIRED);
        this->asmb.jmp_abs(this->dispatch);

        // the stale generation second chance, one per way: reads the primary
        // ITLB entry for the target and, when it is fresh under the current
        // epoch and still names the physical page the way was bound to,
        // restamps the generation and takes the jump without ever leaving
        // generated code. Mac OS X moves the generation thousands of times a
        // second without moving the mappings underneath, and this is what
        // keeps that from unbinding the world each time
        this->emit_va_probe(probe0, thunk, &slot->phys0, &slot->gen0, &slot->code0);
        this->emit_va_probe(probe1, thunk, &slot->phys1, &slot->gen1, &slot->code1);

        this->va_chain_exits.push_back({slot, 0, thunk});
        return true;
    }

    /** The inline revalidation emit_va_chained_exit describes. RDX holds the
        target address and stays untouched for the thunk's sake; RAX and the
        scratch register are free here, nothing being live at an exit */
    void emit_va_probe(X64Emitter::Label probe, X64Emitter::Label thunk,
                       uint32_t* phys, uint64_t* gen, void** code) {
        this->asmb.bind(probe);

        // entry = pCurITLB1 + ((va >> 12) & tlb_size_mask) * sizeof(TLBEntry)
        this->asmb.mov_reg_reg32(RAX, RDX);
        this->asmb.shr_reg_imm8(RAX, PPC_PAGE_SIZE_BITS);
        this->asmb.and_reg_imm32(RAX, TLB_SIZE - 1);
        this->asmb.shl_reg_imm8(RAX, TLB_ENTRY_SHIFT);
        this->asmb.mov_reg_imm64(REG_SCRATCH, uint64_t(uintptr_t(&pCurITLB1)));
        this->asmb.add_reg64_mem(RAX, REG_SCRATCH, 0);

        // fresh under the current epoch?
        this->asmb.mov_reg_reg32(REG_SCRATCH, RDX);
        this->asmb.and_reg_imm32(REG_SCRATCH, PPC_PAGE_MASK);
        this->asmb.or_reg_mem32(REG_SCRATCH, REG_TIME, this->iepoch_disp);
        this->asmb.cmp_mem_reg32(RAX, TLB_TAG_OFFSET, REG_SCRATCH);
        this->asmb.jcc(X64Cond::NotEqual, thunk);

        // still the physical page the binding was made to?
        this->asmb.mov_reg_mem32(REG_SCRATCH, RAX, TLB_PHYS_OFFSET);
        this->asmb.cmp_reg_mem32_abs(REG_SCRATCH, phys);
        this->asmb.jcc(X64Cond::NotEqual, thunk);

        this->asmb.mov_reg64_mem(RAX, REG_TIME, this->gen_disp);
        this->asmb.mov_mem64_abs_reg(gen, RAX);
        this->asmb.jmp_mem_abs(code);
    }

    /** The seam of a cross page walk through: the same primary ITLB probe
        the address predicted chains revalidate with, against an expected
        physical page known at translate time. Passing falls through into
        the callee's instructions; failing leaves through a plain exit with
        pc on the branch target and everything before the seam retired, and
        the dispatcher redoes the fetch honestly. No IR value is live here,
        the builder saw to that */
    void emit_itrans_guard(const IRInsn& in) {
        X64Emitter::Label fail = this->asmb.new_label();
        X64Emitter::Label ok   = this->asmb.new_label();

        // the generation stamp is the hot path: while nothing that affects
        // instruction translation has happened, the seam costs one warm
        // compare instead of a walk into the cold ITLB array. Probing every
        // pass was measured 8% down on the plateau, which is this whole
        // exit's budget several times over
        uint64_t* seam_gen =
            reinterpret_cast<uint64_t*>(this->code.slot_alloc(8, 8));
        if (seam_gen) {
            *seam_gen = 0; // stale until the first pass proves the mapping
            this->asmb.mov_reg64_mem_abs(RAX, seam_gen);
            this->asmb.cmp_reg64_mem(RAX, REG_TIME, this->gen_disp);
            this->asmb.jcc(X64Cond::Equal, ok);
        }

        this->asmb.lea_reg_mem(RDX, REG_ENTRYPC, in.offset);

        // entry = pCurITLB1 + ((va >> 12) & tlb_size_mask) * sizeof(TLBEntry)
        this->asmb.mov_reg_reg32(RAX, RDX);
        this->asmb.shr_reg_imm8(RAX, PPC_PAGE_SIZE_BITS);
        this->asmb.and_reg_imm32(RAX, TLB_SIZE - 1);
        this->asmb.shl_reg_imm8(RAX, TLB_ENTRY_SHIFT);
        this->asmb.mov_reg_imm64(REG_SCRATCH, uint64_t(uintptr_t(&pCurITLB1)));
        this->asmb.add_reg64_mem(RAX, REG_SCRATCH, 0);

        // fresh under the current epoch, and still the physical page the
        // walk through was translated against?
        this->asmb.mov_reg_reg32(REG_SCRATCH, RDX);
        this->asmb.and_reg_imm32(REG_SCRATCH, PPC_PAGE_MASK);
        this->asmb.or_reg_mem32(REG_SCRATCH, REG_TIME, this->iepoch_disp);
        this->asmb.cmp_mem_reg32(RAX, TLB_TAG_OFFSET, REG_SCRATCH);
        this->asmb.jcc(X64Cond::NotEqual, fail);
        this->asmb.mov_reg_mem32(REG_SCRATCH, RAX, TLB_PHYS_OFFSET);
        this->asmb.cmp_reg_imm32(REG_SCRATCH, int32_t(in.imm));
        this->asmb.jcc(X64Cond::NotEqual, fail);

        // proven: remember under which generation, so the next pass skips
        if (seam_gen) {
            this->asmb.mov_reg64_mem(RAX, REG_TIME, this->gen_disp);
            this->asmb.mov_mem64_abs_reg(seam_gen, RAX);
        }
        this->asmb.jmp(ok);

        this->asmb.bind(fail);
        this->emit_exit(in.offset, in.insn_idx - this->accounted);
        this->asmb.bind(ok);
    }

    /** One resolver per chained exit, emitted with the block and pointed at
        by its slot until rt_chain_resolve binds a target. The exit already
        advanced the entry PC register and zeroed the count, so all that is
        left is publishing the PC and asking */
    void emit_chain_thunks() {
        const X64Gpr arg0 = X64Gpr(this->abi.int_arg_regs[0]);
        const X64Gpr ret  = X64Gpr(this->abi.int_ret_reg);

        for (ChainExit& ce : this->chain_exits) {
            ce.thunk_off = this->asmb.size();

            this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, REG_ENTRYPC);
            this->asmb.mov_reg_imm64(arg0, uint64_t(uintptr_t(ce.slot)));
            this->call_absolute(uint64_t(uintptr_t(&rt_chain_resolve)));

            X64Emitter::Label give_up = this->asmb.new_label();
            this->asmb.test_reg64_self(ret);
            this->asmb.jcc(X64Cond::Equal, give_up);
            this->asmb.jmp_reg(ret);

            this->asmb.bind(give_up);
            this->asmb.jmp_abs(this->dispatch);
        }

        for (VaChainExit& ce : this->va_chain_exits) {
            ce.thunk_off = this->asmb.size();
            this->asmb.bind(ce.thunk);

            this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, REG_ENTRYPC);
            this->asmb.mov_reg_imm64(arg0, uint64_t(uintptr_t(ce.slot)));
            this->call_absolute(uint64_t(uintptr_t(&rt_chain_resolve_va)));

            X64Emitter::Label give_up = this->asmb.new_label();
            this->asmb.test_reg64_self(ret);
            this->asmb.jcc(X64Cond::Equal, give_up);
            this->asmb.jmp_reg(ret);

            this->asmb.bind(give_up);
            this->asmb.jmp_abs(this->dispatch);
        }
    }

    /** Records, for every value, the index of the instruction that reads it
        last. That is all a greedy allocator needs when nothing is live across
        a call and there is no control flow inside the block */
    void plan_lifetimes(const IRBlock& ir) {
        this->last_use.assign(ir.insns.size(), 0);
        this->reg_of.assign(ir.insns.size(), NO_REG);
        this->immediate_const.assign(ir.insns.size(), 0);
        this->immediate_value.assign(ir.insns.size(), 0);
        this->dead_gpr_store.assign(ir.insns.size(), 0);

        std::vector<uint16_t> use_count(ir.insns.size(), 0);

        // A later write to the same guest register makes the earlier store
        // unobservable until something can leave generated code or call a
        // helper that reads ppc_state. Keep the IR use in the lifetime plan
        // even when its memory write is dead: consuming it at the same point
        // avoids extending host-register pressure while removing the store.
        IRValue pending_store[32];
        for (IRValue& pending : pending_store) {
            pending = IR_NO_VALUE;
        }
        for (size_t i = 0; i < ir.insns.size(); i++) {
            const IRInsn& in = ir.insns[i];
            const bool barrier = in.opcode == IROpcode::Call ||
                                 in.opcode == IROpcode::Load ||
                                 in.opcode == IROpcode::Store ||
                                 in.opcode == IROpcode::Branch ||
                                 in.opcode == IROpcode::ItransGuard;
            if (barrier) {
                for (IRValue& pending : pending_store) {
                    pending = IR_NO_VALUE;
                }
            }
            if (in.opcode == IROpcode::LoadGPR) {
                // A reload observes this one state slot. The builder normally
                // reaches it only after one of the full barriers above, but
                // keeping the IR rule local makes the pass safe if that cache
                // policy changes later.
                pending_store[in.reg] = IR_NO_VALUE;
            }
            if (in.opcode == IROpcode::StoreGPR) {
                if (pending_store[in.reg] != IR_NO_VALUE) {
                    this->dead_gpr_store[pending_store[in.reg]] = 1;
                }
                pending_store[in.reg] = IRValue(i);
            }
        }

        for (size_t i = 0; i < ir.insns.size(); i++) {
            const IRInsn& in = ir.insns[i];
            if (in.a != IR_NO_VALUE) {
                this->last_use[in.a] = uint32_t(i);
                use_count[in.a]++;
            }
            if (in.b != IR_NO_VALUE) {
                this->last_use[in.b] = uint32_t(i);
                use_count[in.b]++;
            }
        }

        // x86 can consume one constant operand without first materialising
        // it in a host register. Restrict this to a single-use value and one
        // operand per instruction: every skipped ConstI32 then has exactly
        // one immediate consumer, while all other backends keep seeing the
        // ordinary, backend-neutral IR.
        for (size_t i = 0; i < ir.insns.size(); i++) {
            const IRInsn& in = ir.insns[i];
            const bool immediate_alu = !in.oe &&
                (in.opcode == IROpcode::Add || in.opcode == IROpcode::And ||
                 in.opcode == IROpcode::Or  || in.opcode == IROpcode::Xor);
            if (!immediate_alu) {
                continue;
            }

            IRValue selected = IR_NO_VALUE;
            if (in.b != IR_NO_VALUE && use_count[in.b] == 1 &&
                ir.insns[in.b].opcode == IROpcode::ConstI32) {
                selected = in.b;
            } else if (in.a != IR_NO_VALUE && use_count[in.a] == 1 &&
                       ir.insns[in.a].opcode == IROpcode::ConstI32) {
                selected = in.a;
            }
            if (selected != IR_NO_VALUE) {
                this->immediate_const[selected] = 1;
                this->immediate_value[selected] = ir.insns[selected].imm;
            }
        }
    }

    bool takes_register(IROpcode op) const {
        switch (op) {
        case IROpcode::LoadGPR:
        case IROpcode::LoadSPR:
        case IROpcode::ConstI32:
        case IROpcode::PcRel:
        case IROpcode::Add:
        case IROpcode::Sub:
        case IROpcode::And:
        case IROpcode::Or:
        case IROpcode::Xor:
        case IROpcode::RotlMask:
        case IROpcode::Exts:
        case IROpcode::MulLow:
        case IROpcode::MulHighS:
        case IROpcode::MulHighU:
            return true;
        default:
            return false;
        }
    }

    bool emit_value_op(const IRInsn& in, size_t idx, uint32_t& free_mask) {
        const bool a_dies = in.a != IR_NO_VALUE && this->last_use[in.a] == idx;
        const bool b_dies = in.b != IR_NO_VALUE && this->last_use[in.b] == idx;

        // A selected single-use constant is emitted by its consumer below.
        // Giving it no register removes both the mov-immediate and its
        // register pressure from the generated block.
        if (in.opcode == IROpcode::ConstI32 && this->immediate_const[idx]) {
            return true;
        }

        X64Gpr ra = in.a != IR_NO_VALUE ? X64Gpr(this->reg_of[in.a]) : RAX;
        X64Gpr rb = in.b != IR_NO_VALUE ? X64Gpr(this->reg_of[in.b]) : RAX;

        if (in.opcode == IROpcode::StoreGPR) {
            if (!this->dead_gpr_store[idx]) {
                this->asmb.mov_mem_reg32(REG_STATE, gpr_offset(in.reg), ra);
            }
            if (a_dies) free_mask |= bit(uint8_t(ra));
            return true;
        }

        if (in.opcode == IROpcode::StoreSPR) {
            this->asmb.mov_mem_reg32(REG_STATE, spr_offset(in.reg), ra);
            if (a_dies) free_mask |= bit(uint8_t(ra));
            return true;
        }

        if (!this->takes_register(in.opcode)) {
            return false; // Phi and anything else added later
        }

        const IRValue imm_value =
            in.a != IR_NO_VALUE && this->immediate_const[in.a] ? in.a :
            in.b != IR_NO_VALUE && this->immediate_const[in.b] ? in.b : IR_NO_VALUE;
        if (imm_value != IR_NO_VALUE) {
            const IRValue src_value = in.a == imm_value ? in.b : in.a;
            const X64Gpr src = X64Gpr(this->reg_of[src_value]);
            const bool src_dies = this->last_use[src_value] == idx;

            X64Gpr dst;
            if (src_dies) {
                dst = src;
            } else {
                if (!free_mask) {
                    LOG_F(INFO, "JIT: out of host registers, block goes to the interpreter");
                    return false;
                }
                dst = X64Gpr(lowest_bit(free_mask));
                free_mask &= ~bit(uint8_t(dst));
                this->asmb.mov_reg_reg32(dst, src);
            }

            const uint32_t imm = this->immediate_value[imm_value];
            switch (in.opcode) {
            case IROpcode::Add: this->asmb.add_reg_imm32(dst, imm); break;
            case IROpcode::And: this->asmb.and_reg_imm32(dst, imm); break;
            case IROpcode::Or:  this->asmb.or_reg_imm32(dst, imm);  break;
            case IROpcode::Xor: this->asmb.xor_reg_imm32(dst, imm); break;
            default: return false;
            }

            this->reg_of[idx] = uint8_t(dst);
            return true;
        }

        // reusing a dying operand's register saves the move that would
        // otherwise start every two operand form
        X64Gpr dst;
        if (a_dies) {
            dst = ra;
        } else if (b_dies && is_commutative(in.opcode)) {
            dst = rb;
        } else {
            if (!free_mask) {
                LOG_F(INFO, "JIT: out of host registers, block goes to the interpreter");
                return false;
            }
            dst = X64Gpr(lowest_bit(free_mask));
            free_mask &= ~bit(uint8_t(dst));
        }

        switch (in.opcode) {
        case IROpcode::LoadGPR:
            this->asmb.mov_reg_mem32(dst, REG_STATE, gpr_offset(in.reg));
            break;
        case IROpcode::LoadSPR:
            this->asmb.mov_reg_mem32(dst, REG_STATE, spr_offset(in.reg));
            break;
        case IROpcode::ConstI32:
            this->asmb.mov_reg_imm32(dst, in.imm);
            break;
        case IROpcode::PcRel:
            // same shape emit_branch writes LR with: the entry register
            // holds the address this run actually came in through
            this->asmb.lea_reg_mem(dst, REG_ENTRYPC, int32_t(in.imm));
            break;
        case IROpcode::Add:
        case IROpcode::And:
        case IROpcode::Or:
        case IROpcode::Xor:
        case IROpcode::MulLow:
            this->emit_commutative(in.opcode, dst, ra, rb);
            break;
        case IROpcode::Sub:
            // not commutative, so dst never lands on b. When both operands
            // share a register the subtraction still reads them before
            // writing, which makes the aliased case come out right
            if (dst != ra) this->asmb.mov_reg_reg32(dst, ra);
            this->asmb.sub_reg_reg32(dst, rb);
            break;
        case IROpcode::MulHighS: {
            // both operands sign extended to 64 bits, full product, top half.
            // The scratch copy goes first: when dst aliases an operand the
            // other one has to be read before dst is overwritten
            const X64Gpr self  = (dst == rb) ? rb : ra;
            const X64Gpr other = (dst == rb) ? ra : rb;
            this->asmb.movsxd_reg64_reg32(REG_SCRATCH, other);
            this->asmb.movsxd_reg64_reg32(dst, self);
            this->asmb.imul_reg64_reg64(dst, REG_SCRATCH);
            this->asmb.shr_reg64_imm8(dst, 32);
            break;
        }
        case IROpcode::MulHighU: {
            // every I32 value leaves its upper half zero, since each producer
            // is a 32 bit operation, so the operands are already the zero
            // extended 64 bit values the unsigned product needs
            const X64Gpr self  = (dst == rb) ? rb : ra;
            const X64Gpr other = (dst == rb) ? ra : rb;
            if (dst != self) this->asmb.mov_reg_reg32(dst, self);
            this->asmb.imul_reg64_reg64(dst, other);
            this->asmb.shr_reg64_imm8(dst, 32);
            break;
        }
        case IROpcode::RotlMask:
            if (dst != ra) this->asmb.mov_reg_reg32(dst, ra);
            this->asmb.rol_reg_imm8(dst, in.sh);
            this->asmb.and_reg_imm32(dst, rot_mask(in.mb, in.me));
            break;
        case IROpcode::Exts:
            if (in.width == 1) this->asmb.movsx_reg8(dst, ra);
            else               this->asmb.movsx_reg16(dst, ra);
            break;
        default:
            return false;
        }

        // free whatever died and was not taken over by the destination
        if (a_dies && ra != dst) free_mask |= bit(uint8_t(ra));
        if (b_dies && rb != dst) free_mask |= bit(uint8_t(rb));

        this->reg_of[idx] = uint8_t(dst);
        return true;
    }

    static bool flags_survive(IROpcode op) {
        return op == IROpcode::StoreGPR || op == IROpcode::StoreSPR ||
               op == IROpcode::LoadGPR  || op == IROpcode::LoadSPR  ||
               op == IROpcode::ConstI32 || op == IROpcode::PcRel    ||
               op == IROpcode::Exts;
    }

    static bool is_commutative(IROpcode op) {
        return op == IROpcode::Add      || op == IROpcode::And ||
               op == IROpcode::Or       || op == IROpcode::Xor ||
               op == IROpcode::MulLow   || op == IROpcode::MulHighS ||
               op == IROpcode::MulHighU;
    }

    /** The XER[CA] family: the operation itself, then the host carry flag
        folded into XER. The host flag is the whole trick: add, adc and sub
        leave in CF exactly the carry the formulas in jitir.h describe, so the
        arithmetic costs what a plain Add does and the only extra work is the
        read-modify-write of XER.

        The scratch register carries the carry in and out, since nothing else
        may touch the flags between the operation and the setcc. One more
        register is needed for the XER merge; a block too pressured to have
        one goes to the interpreter like any other allocation failure */
    bool emit_carry(const IRInsn& in, size_t idx, uint32_t& free_mask) {
        const X64Gpr ra = X64Gpr(this->reg_of[in.a]);
        const X64Gpr rb = X64Gpr(this->reg_of[in.b]);
        const bool a_dies = this->last_use[in.a] == idx;
        const bool b_dies = this->last_use[in.b] == idx;

        const bool wants_cf = in.opcode == IROpcode::AddECA ||
                              in.opcode == IROpcode::SubECA;
        const bool commutative = in.opcode == IROpcode::AddCA ||
                                 in.opcode == IROpcode::AddECA;

        // same reuse rules as emit_value_op, with one exception: SubECA flips
        // dst before reading b, so the two may not share a register, which
        // aliased operands would otherwise make happen
        X64Gpr dst;
        if (a_dies && !(in.opcode == IROpcode::SubECA && ra == rb)) {
            dst = ra;
        } else if (b_dies && commutative) {
            dst = rb;
        } else {
            if (!free_mask) {
                return false;
            }
            dst = X64Gpr(lowest_bit(free_mask));
            free_mask &= ~bit(uint8_t(dst));
        }

        uint32_t avail = free_mask & ~bit(uint8_t(dst)) &
                         ~bit(uint8_t(ra)) & ~bit(uint8_t(rb));
        if (!avail) {
            return false;
        }
        const X64Gpr rtmp = X64Gpr(lowest_bit(avail));

        if (wants_cf) {
            // CA sits in bit 29, so shifting by 30 drops it into CF
            this->asmb.mov_reg_mem32(REG_SCRATCH, REG_STATE, XER_OFFSET);
            this->asmb.shr_reg_imm8(REG_SCRATCH, 30);
        }

        switch (in.opcode) {
        case IROpcode::AddCA:
            if (dst == rb)      this->asmb.add_reg_reg32(dst, ra);
            else {
                if (dst != ra)  this->asmb.mov_reg_reg32(dst, ra);
                this->asmb.add_reg_reg32(dst, rb);
            }
            break;
        case IROpcode::AddECA:
            // mov does not touch the flags, so the carry survives it
            if (dst == rb)      this->asmb.adc_reg_reg32(dst, ra);
            else {
                if (dst != ra)  this->asmb.mov_reg_reg32(dst, ra);
                this->asmb.adc_reg_reg32(dst, rb);
            }
            break;
        case IROpcode::SubCA:
            if (dst != ra)      this->asmb.mov_reg_reg32(dst, ra);
            this->asmb.sub_reg_reg32(dst, rb);
            break;
        default: // SubECA, ~a + b + CA; not leaves the flags alone
            if (dst != ra)      this->asmb.mov_reg_reg32(dst, ra);
            this->asmb.not_reg32(dst);
            this->asmb.adc_reg_reg32(dst, rb);
            break;
        }

        // SubCA wants no-borrow, everything else wants the carry as it is.
        // Both setcc go first, before anything can disturb the flags; the OE
        // forms fold the overflow in as SO|OV set together and OV cleared
        // alone, which is what ppc_setsoov does. The host OF is that overflow
        // for every shape here, carry in included
        this->asmb.setcc_reg8(in.opcode == IROpcode::SubCA ? X64Cond::AboveEqual
                                                           : X64Cond::Below,
                              REG_SCRATCH);
        if (in.oe) {
            this->asmb.setcc_reg8(X64Cond::Overflow, rtmp);
        }
        this->asmb.movzx_reg8(REG_SCRATCH, REG_SCRATCH);
        this->asmb.shl_reg_imm8(REG_SCRATCH, 29);
        if (in.oe) {
            this->asmb.movzx_reg8(rtmp, rtmp);
            this->asmb.neg_reg32(rtmp); // all ones when it overflowed
            this->asmb.and_reg_imm32(rtmp, XER::SO | XER::OV);
            this->asmb.or_reg_reg32(REG_SCRATCH, rtmp);
        }
        this->asmb.mov_reg_mem32(rtmp, REG_STATE, XER_OFFSET);
        this->asmb.and_reg_imm32(rtmp, in.oe ? ~uint32_t(XER::CA | XER::OV)
                                             : ~uint32_t(XER::CA));
        this->asmb.or_reg_reg32(rtmp, REG_SCRATCH);
        this->asmb.mov_mem_reg32(REG_STATE, XER_OFFSET, rtmp);

        if (a_dies && ra != dst) free_mask |= bit(uint8_t(ra));
        if (b_dies && rb != dst) free_mask |= bit(uint8_t(rb));
        this->reg_of[idx] = uint8_t(dst);
        return true;
    }

    /** The OE forms outside the carry family: addo, subfo, nego and mullwo.
        The operation is the plain one, the host OF is the guest overflow, and
        the XER merge is the same shape emit_carry uses minus the CA bit */
    bool emit_ov_op(const IRInsn& in, size_t idx, uint32_t& free_mask) {
        const X64Gpr ra = X64Gpr(this->reg_of[in.a]);
        const X64Gpr rb = X64Gpr(this->reg_of[in.b]);
        const bool a_dies = this->last_use[in.a] == idx;
        const bool b_dies = this->last_use[in.b] == idx;
        const bool commutative = in.opcode != IROpcode::Sub;

        X64Gpr dst;
        if (a_dies) {
            dst = ra;
        } else if (b_dies && commutative) {
            dst = rb;
        } else {
            if (!free_mask) {
                return false;
            }
            dst = X64Gpr(lowest_bit(free_mask));
            free_mask &= ~bit(uint8_t(dst));
        }

        const uint32_t avail = free_mask & ~bit(uint8_t(dst)) &
                               ~bit(uint8_t(ra)) & ~bit(uint8_t(rb));
        if (!avail) {
            return false;
        }
        const X64Gpr rtmp = X64Gpr(lowest_bit(avail));

        if (in.opcode == IROpcode::Sub) {
            if (dst != ra) this->asmb.mov_reg_reg32(dst, ra);
            this->asmb.sub_reg_reg32(dst, rb);
        } else {
            this->emit_commutative(in.opcode, dst, ra, rb);
        }

        this->asmb.setcc_reg8(X64Cond::Overflow, REG_SCRATCH);
        this->asmb.movzx_reg8(REG_SCRATCH, REG_SCRATCH);
        this->asmb.neg_reg32(REG_SCRATCH);
        this->asmb.and_reg_imm32(REG_SCRATCH, XER::SO | XER::OV);
        this->asmb.mov_reg_mem32(rtmp, REG_STATE, XER_OFFSET);
        this->asmb.and_reg_imm32(rtmp, ~uint32_t(XER::OV));
        this->asmb.or_reg_reg32(rtmp, REG_SCRATCH);
        this->asmb.mov_mem_reg32(REG_STATE, XER_OFFSET, rtmp);

        if (a_dies && ra != dst) free_mask |= bit(uint8_t(ra));
        if (b_dies && rb != dst) free_mask |= bit(uint8_t(rb));
        this->reg_of[idx] = uint8_t(dst);
        return true;
    }

    /** mtcrf. The full mask is one store; anything partial merges under a
        mask that has been a constant since translation */
    bool emit_mtcrf(const IRInsn& in, size_t idx, uint32_t& free_mask) {
        const X64Gpr ra = X64Gpr(this->reg_of[in.a]);
        const bool a_dies = this->last_use[in.a] == idx;

        if (in.imm == 0xFFFFFFFFUL) {
            this->asmb.mov_mem_reg32(REG_STATE, CR_OFFSET, ra);
        } else if (in.imm != 0) { // a zero CRM writes nothing at all
            const uint32_t avail = free_mask & ~bit(uint8_t(ra));
            if (!avail) {
                return false;
            }
            const X64Gpr rtmp = X64Gpr(lowest_bit(avail));
            this->asmb.mov_reg_reg32(REG_SCRATCH, ra);
            this->asmb.and_reg_imm32(REG_SCRATCH, in.imm);
            this->asmb.mov_reg_mem32(rtmp, REG_STATE, CR_OFFSET);
            this->asmb.and_reg_imm32(rtmp, ~in.imm);
            this->asmb.or_reg_reg32(rtmp, REG_SCRATCH);
            this->asmb.mov_mem_reg32(REG_STATE, CR_OFFSET, rtmp);
        }

        if (a_dies) free_mask |= bit(uint8_t(ra));
        return true;
    }

    /** Every op here is commutative, so when the destination landed on the
        second operand the operands simply swap instead of needing a temporary */
    void emit_commutative(IROpcode op, X64Gpr dst, X64Gpr a, X64Gpr b) {
        X64Gpr other;
        if (dst == a) {
            other = b;
        } else if (dst == b) {
            other = a;
        } else {
            this->asmb.mov_reg_reg32(dst, a);
            other = b;
        }
        switch (op) {
        case IROpcode::Add:    this->asmb.add_reg_reg32(dst, other);  break;
        case IROpcode::And:    this->asmb.and_reg_reg32(dst, other);  break;
        case IROpcode::Or:     this->asmb.or_reg_reg32(dst, other);   break;
        case IROpcode::MulLow: this->asmb.imul_reg_reg32(dst, other); break;
        default:               this->asmb.xor_reg_reg32(dst, other);  break;
        }
    }

    static uint32_t rot_mask(unsigned mb, unsigned me) {
        uint32_t begin = 0xFFFFFFFFUL >> mb;
        uint32_t end   = me >= 31 ? 0 : 0xFFFFFFFFUL >> (me + 1);
        uint32_t mask  = begin ^ end;
        return (me < mb) ? ~mask : mask;
    }

    static uint8_t lowest_bit(uint32_t mask) {
        for (uint8_t i = 0; i < 32; i++) {
            if (mask & (1u << i)) return i;
        }
        return NO_REG;
    }

    /** The two stubs every block shares, emitted once ahead of everything and
        kept below the code memory floor so a flush cannot take them away.

        They own the frame. The trampoline is called from C++ and jumps into a
        block; the dispatch stub is jumped to by a block and either jumps into
        the next one or unwinds and returns. Blocks in between are pure body */
    bool emit_shared_stubs() {
        if (!this->code.begin_write()) {
            return false;
        }

        const bool ok = this->emit_trampoline() && this->emit_dispatch();

        if (!this->code.end_write() || !ok) {
            return false;
        }

        // everything above this point survives release_all
        this->code.mark_floor();
        return true;
    }

    bool emit_trampoline() {
        const X64Gpr arg0 = X64Gpr(this->abi.int_arg_regs[0]);

        this->asmb.clear();

        for (size_t i = 0; i < NUM_PINNED; i++) {
            this->asmb.push(pinned[i]);
        }
        this->asmb.sub_rsp(this->pad);

        // the block pointer is in a volatile register, so read what we need
        // from it before anything else can want that register
        this->asmb.mov_reg64_mem(REG_SCRATCH, arg0, int32_t(offsetof(JitBlock, code)));

        this->asmb.mov_reg_imm64(REG_STATE, uint64_t(uintptr_t(&ppc_state)));
        this->asmb.mov_reg_mem32(REG_ENTRYPC, REG_STATE, PC_OFFSET);
        this->asmb.mov_reg_imm64(REG_FLAGS, uint64_t(uintptr_t(&exec_flags)));
        this->asmb.mov_reg_imm64(REG_CALLOP, uint64_t(uintptr_t(&rt_call_op)));
        this->asmb.xor_reg_reg32(REG_RETIRED, REG_RETIRED);

        this->asmb.jmp_reg(REG_SCRATCH);

        uint8_t* dst = this->place_stub();
        this->trampoline = reinterpret_cast<BlockEntry>(dst);
        return dst != nullptr;
    }

    bool emit_dispatch() {
        const X64Gpr arg0 = X64Gpr(this->abi.int_arg_regs[0]);
        const X64Gpr ret  = X64Gpr(this->abi.int_ret_reg);

        this->asmb.clear();

        const X64Emitter::Label leave = this->asmb.new_label();

        this->asmb.mov_reg_reg32(arg0, REG_RETIRED);
        this->call_absolute(uint64_t(uintptr_t(&rt_dispatch)));

        this->asmb.test_reg64_self(ret);
        this->asmb.jcc(X64Cond::Equal, leave);

        // straight into the next block, still on this frame. Nothing may
        // touch the return register between here and the jump
        this->asmb.mov_reg_mem32(REG_ENTRYPC, REG_STATE, PC_OFFSET);
        this->asmb.xor_reg_reg32(REG_RETIRED, REG_RETIRED);
        this->asmb.jmp_reg(ret);

        this->asmb.bind(leave);
        this->asmb.add_rsp(this->pad);
        for (size_t i = NUM_PINNED; i-- > 0;) {
            this->asmb.pop(pinned[i]);
        }
        this->asmb.ret(); // back to whoever called the trampoline

        uint8_t* dst = this->place_stub();
        this->dispatch = dst;
        return dst != nullptr;
    }

    /** Copies what the emitter is holding into code memory. Only for the
        stubs, which are emitted while the region is already writable */
    uint8_t* place_stub() {
        if (!this->asmb.finalize()) {
            return nullptr;
        }
        uint8_t* dst = this->code.alloc(this->asmb.size());
        if (!dst) {
            return nullptr;
        }
        std::memcpy(dst, this->asmb.bytes(), this->asmb.size());
        return this->asmb.relocate(dst) ? dst : nullptr;
    }

    bool emit_call(const IRInsn& in, X64Gpr arg0, X64Gpr arg1) {
        const uint32_t guest_idx = in.insn_idx;

        // ppc_state.pc = entry_pc + offset, which helpers read for branch
        // targets and for the address an exception records. It goes first
        // because the cycle settling below needs it too
        if (in.offset == 0) {
            this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, REG_ENTRYPC);
        } else {
            this->asmb.lea_reg_mem(arg0, REG_ENTRYPC, int32_t(in.offset));
            this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, arg0);
        }

        if (!this->emit_cycle_sync(guest_idx)) {
            return false;
        }

        this->asmb.mov_reg_imm64(arg0, uint64_t(uintptr_t(in.helper)));
        this->asmb.mov_reg_imm32(arg1, in.imm);
        this->asmb.call_reg(REG_CALLOP);

        // The hot path leaves nothing behind: exec_flags is the only thing
        // worth testing, because rt_call_op only returns false by way of
        // ppc_exception_handler_unwind, which runs the handler before it
        // throws. Both ways out therefore raise a flag, and everything that
        // tells them apart is bookkeeping nobody reads unless we leave, so it
        // goes out of line
        X64Emitter::Label cold = this->asmb.new_label();
        this->cold_exits.push_back({cold, guest_idx - this->accounted});

        this->asmb.cmp_mem_imm8(REG_FLAGS, 0, 0);
        this->asmb.jcc(X64Cond::NotEqual, cold);

        return true;
    }

    /** Adds the cycle prefix a path still owes and lets timers run if their
        deadline has passed.

        A block used to do this only when it ended, which meant an instruction
        in the middle saw virtual time from before the block started. Every
        device the emulator has derives its timing from that counter, so a
        helper reaching one got answers from the past and a booting system
        walked off a jump table. The cost is six instructions ahead of a call
        that already costs twenty.

        ppc_state.pc must already name the instruction about to run: it has
        not run, and that is where an exception delivered here resumes. The
        caller decides whether this prefix belongs to every runtime path and
        therefore changes `accounted`, or to a slow path that exits here. */
    bool emit_cycle_sync_owed(uint32_t owed) {
        X64Emitter::Label service = this->asmb.new_label();
        X64Emitter::Label done    = this->asmb.new_label();

        if (owed) {
            this->asmb.add_mem64_imm32(REG_TIME, this->icycles_disp, owed);
        }

        this->asmb.mov_reg64_mem(RAX, REG_TIME, this->icycles_disp);
        this->asmb.cmp_reg64_mem(RAX, REG_TIME, this->deadline_disp);
        this->asmb.jcc(X64Cond::Above, service);
        this->asmb.cmp_mem8_imm8(REG_TIME, this->timer_disp, 0);
        this->asmb.jcc(X64Cond::Equal, done);

        this->asmb.bind(service);
        this->call_absolute(uint64_t(uintptr_t(&rt_service_timers)));
        this->asmb.test_reg8_self(RAX);

        // false means a timer raised something, so the instruction never ran
        // and nothing more of this block is owed
        X64Emitter::Label leave = this->asmb.new_label();
        this->sync_exits.push_back(leave);
        this->asmb.jcc(X64Cond::Equal, leave);

        this->asmb.bind(done);
        return true;
    }

    bool emit_cycle_sync(uint32_t guest_idx) {
        const uint32_t owed = guest_idx - this->accounted;
        this->accounted     = guest_idx;
        return this->emit_cycle_sync_owed(owed);
    }

    /** Runs a memory instruction through its interpreter helper after the
        inline TLB path declined it. MMIO and every other slow access can
        observe virtual time, so the cycles preceding the instruction have
        to be visible first.

        This path always leaves through dispatch. Rejoining the fast path
        would make `accounted` describe two different runtime histories: the
        slow path settled its prefix while the fast one did not. Dispatching
        also keeps a device access that changed mappings or executable pages
        from running the rest of a translation whose assumptions just moved. */
    bool emit_memory_slow_call(const IRInsn& in, X64Gpr pc_tmp,
                               X64Gpr arg0, X64Gpr arg1)
    {
        this->asmb.lea_reg_mem(pc_tmp, REG_ENTRYPC, int32_t(in.offset));
        this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, pc_tmp);

        if (!this->emit_cycle_sync_owed(in.insn_idx - this->accounted)) {
            return false;
        }

        this->asmb.mov_reg_imm64(arg0, uint64_t(uintptr_t(in.helper)));
        this->asmb.mov_reg_imm32(arg1, in.imm);
        this->asmb.call_reg(REG_CALLOP);

        // The prefix was settled above, so this helper contributes only its
        // own retirement: one when it returned and raised, zero when it
        // unwound. emit_cold_exits picks that bit out of the return register.
        X64Emitter::Label cold = this->asmb.new_label();
        this->cold_exits.push_back({cold, 0});
        this->asmb.cmp_mem_imm8(REG_FLAGS, 0, 0);
        this->asmb.jcc(X64Cond::NotEqual, cold);

        // A completed load/store retires here. rt_dispatch accounts that one
        // instruction and resumes at the following PC.
        this->asmb.lea_reg_mem(pc_tmp, REG_ENTRYPC, int32_t(in.offset) + 4);
        this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, pc_tmp);
        this->asmb.mov_reg_imm32(REG_RETIRED, 1);
        this->asmb.jmp_abs(this->dispatch);
        return true;
    }

    /** A guest load, with the primary TLB hit inlined.

        The fast path is a tag compare and an alignment test, plus the rA
        writeback of the u forms, whose address it already holds. Anything
        else runs the interpreter's own helper for the instruction, whole:
        effective address, access, rd and any update in one piece. The whole
        piece matters. A device read can raise an exception and still
        complete, and a version that did only the memory access out of line
        would leave rd unwritten while the guest, sent back to the same
        instruction, read the device a second time.

        A slow access leaves through dispatch after the helper. Only the
        inline memory path reaches the SSA continuation below.

        The translator drops the register cache at every load, so the address
        is the one value live at this point and the call is free to clobber
        the rest of the pool */
    bool emit_load(const IRInsn& in, size_t idx, uint32_t& free_mask,
                   X64Gpr arg0, X64Gpr arg1, X64Gpr ret)
    {
        (void)ret;
        const bool ea_dies = this->last_use[in.a] == idx;
        const X64Gpr rea   = X64Gpr(this->reg_of[in.a]);

        // the address is read after the destination is written, so unlike the
        // arithmetic forms the destination may not take it over
        uint32_t avail = free_mask & ~bit(uint8_t(rea));
        if (!avail) {
            return false;
        }
        const X64Gpr rdst = X64Gpr(lowest_bit(avail));
        avail &= ~bit(uint8_t(rdst));
        if (!avail) {
            return false;
        }
        const X64Gpr rtmp = X64Gpr(lowest_bit(avail));

        X64Emitter::Label slow = this->asmb.new_label();
        X64Emitter::Label done = this->asmb.new_label();

        // entry = pCurDTLB1 + ((ea >> 12) & tlb_size_mask) * sizeof(TLBEntry)
        this->asmb.mov_reg_reg32(rtmp, rea);
        this->asmb.shr_reg_imm8(rtmp, PPC_PAGE_SIZE_BITS);
        this->asmb.and_reg_imm32(rtmp, TLB_SIZE - 1);
        this->asmb.shl_reg_imm8(rtmp, TLB_ENTRY_SHIFT);
        this->asmb.mov_reg_imm64(REG_SCRATCH, uint64_t(uintptr_t(&pCurDTLB1)));
        this->asmb.add_reg64_mem(rtmp, REG_SCRATCH, 0);

        // entry->tag == (ea & ~0xFFF) | dtlb_epoch ? An entry filled under an
        // older epoch mismatches by construction, which is the whole flush
        this->asmb.mov_reg_reg32(rdst, rea);
        this->asmb.and_reg_imm32(rdst, PPC_PAGE_MASK);
        this->asmb.or_reg_mem32(rdst, REG_TIME, this->depoch_disp);
        this->asmb.cmp_mem_reg32(rtmp, TLB_TAG_OFFSET, rdst);
        this->asmb.jcc(X64Cond::NotEqual, slow);

        if (in.width > 1) {
            this->asmb.test_reg_imm32(rea, in.width - 1);
            this->asmb.jcc(X64Cond::NotEqual, slow);
        }

        // host address = entry->host_va_offs_r + ea, with ea zero extended
        // because every 32 bit write above cleared the upper half
        this->asmb.mov_reg64_mem(rtmp, rtmp, TLB_HOSTR_OFFSET);
        this->asmb.add_reg64_reg64(rtmp, rea);

        switch (in.width) {
        case 1:
            this->asmb.movzx_reg8_mem(rdst, rtmp, 0);
            break;
        case 2:
            // bswap works on the whole 32 bit register, so the halfword lands
            // in the top half and comes back down. A rol of 8 would only be
            // right on a 16 bit operand and silently produces 0x00DDCC00
            // where 0x0000CCDD was wanted. lhbrx wants the bytes as the host
            // reads them, so it is the raw movzx and nothing else
            this->asmb.movzx_reg16_mem(rdst, rtmp, 0);
            if (!in.byte_reverse) {
                this->asmb.bswap_reg32(rdst);
                this->asmb.shr_reg_imm8(rdst, 16);
                if (in.signed_load) {
                    this->asmb.movsx_reg16(rdst, rdst);
                }
            }
            break;
        default:
            // same story: lwbrx on a big endian guest is the host's own order
            this->asmb.mov_reg_mem32(rdst, rtmp, 0);
            if (!in.byte_reverse) {
                this->asmb.bswap_reg32(rdst);
            }
            break;
        }

        // the u forms write the address back to rA. After the value: a fast
        // path access cannot fault, but keeping the interpreter's order costs
        // nothing. The forms where rA overlaps rD were declined at decode
        if (in.ureg != IR_NO_UPDATE) {
            this->asmb.mov_mem_reg32(REG_STATE, gpr_offset(in.ureg), rea);
        }
        this->asmb.jmp(done);

        this->asmb.bind(slow);
        if (!this->emit_memory_slow_call(in, rtmp, arg0, arg1)) {
            return false;
        }

        this->asmb.bind(done);

        if (ea_dies) {
            free_mask |= bit(uint8_t(rea));
        }
        free_mask &= ~bit(uint8_t(rdst));
        this->reg_of[idx] = uint8_t(rdst);
        return true;
    }

    /** Writes one condition register field.

        The three ordering bits are picked with cmov rather than branches, so
        the sequence is straight line and the comparison feeds it directly.
        XER[SO] is copied down into the low bit of the field, then the whole
        thing is shifted into place and merged, which is what ppc_cmpi and
        ppc_changecrf0 do between them */
    bool emit_set_cr(const IRInsn& in, size_t idx, uint32_t& free_mask) {
        const X64Gpr ra = X64Gpr(this->reg_of[in.a]);
        const X64Gpr rb = X64Gpr(this->reg_of[in.b]);
        const bool a_dies = this->last_use[in.a] == idx;
        const bool b_dies = this->last_use[in.b] == idx;

        uint32_t avail = free_mask & ~bit(uint8_t(ra)) & ~bit(uint8_t(rb));
        if (!avail) {
            return false;
        }
        const X64Gpr rfield = X64Gpr(lowest_bit(avail));
        avail &= ~bit(uint8_t(rfield));
        if (!avail) {
            return false;
        }
        const X64Gpr rtmp = X64Gpr(lowest_bit(avail));
        avail &= ~bit(uint8_t(rtmp));

        const X64Cond gt = in.cr_signed ? X64Cond::Greater : X64Cond::Above;
        const X64Cond lt = in.cr_signed ? X64Cond::Less    : X64Cond::Below;

        if (avail) {
            // fusible ordering, one scratch register more: everything that
            // scratches FLAGS runs before the compare, and after it only
            // mov, cmov and lea touch the field, so the compare's FLAGS
            // survive into whatever follows. A branch on this field can
            // then jump straight off them instead of reloading CR
            const X64Gpr rcr = X64Gpr(lowest_bit(avail));

            // XER[SO] sits three bits above where CR wants it
            this->asmb.mov_reg_mem32(rcr, REG_STATE, XER_OFFSET);
            this->asmb.and_reg_imm32(rcr, XER::SO);
            this->asmb.shr_reg_imm8(rcr, 3 + in.crf);
            this->asmb.mov_reg_mem32(rtmp, REG_STATE, CR_OFFSET);
            this->asmb.and_reg_imm32(rtmp, ~(0xF0000000UL >> in.crf));
            // the pieces are disjoint, so the adds below are the or
            this->asmb.lea_reg_reg32(rcr, rcr, rtmp);

            this->asmb.cmp_reg_reg32(ra, rb);

            this->asmb.mov_reg_imm32(rfield, CRx_bit::CR_EQ >> in.crf);
            this->asmb.mov_reg_imm32(rtmp, CRx_bit::CR_GT >> in.crf);
            this->asmb.cmov_reg_reg32(gt, rfield, rtmp);
            this->asmb.mov_reg_imm32(rtmp, CRx_bit::CR_LT >> in.crf);
            this->asmb.cmov_reg_reg32(lt, rfield, rtmp);
            this->asmb.lea_reg_reg32(rfield, rfield, rcr);
            this->asmb.mov_mem_reg32(REG_STATE, CR_OFFSET, rfield);

            this->live_cmp.valid = jit_cr_fuse;
            this->live_cmp.sig   = in.cr_signed;
            this->live_cmp.crf   = in.crf;
        } else {
            // two scratch registers: the legacy ordering, FLAGS do not
            // survive it
            this->asmb.cmp_reg_reg32(ra, rb);

            this->asmb.mov_reg_imm32(rfield, CRx_bit::CR_EQ);
            this->asmb.mov_reg_imm32(rtmp, CRx_bit::CR_GT);
            this->asmb.cmov_reg_reg32(gt, rfield, rtmp);
            this->asmb.mov_reg_imm32(rtmp, CRx_bit::CR_LT);
            this->asmb.cmov_reg_reg32(lt, rfield, rtmp);

            // XER[SO] sits three bits above where CR wants it
            this->asmb.mov_reg_mem32(rtmp, REG_STATE, XER_OFFSET);
            this->asmb.and_reg_imm32(rtmp, XER::SO);
            this->asmb.shr_reg_imm8(rtmp, 3);
            this->asmb.or_reg_reg32(rfield, rtmp);

            this->asmb.shr_reg_imm8(rfield, in.crf);
            this->asmb.mov_reg_mem32(rtmp, REG_STATE, CR_OFFSET);
            this->asmb.and_reg_imm32(rtmp, ~(0xF0000000UL >> in.crf));
            this->asmb.or_reg_reg32(rtmp, rfield);
            this->asmb.mov_mem_reg32(REG_STATE, CR_OFFSET, rtmp);

            this->live_cmp.valid = false;
        }

        if (a_dies) free_mask |= bit(uint8_t(ra));
        if (b_dies) free_mask |= bit(uint8_t(rb));
        return true;
    }

    /** A guest store, with the primary TLB hit inlined.

        Same shape as emit_load with one more gate: the entry has to be
        writable and to have had the PTE change bit set already. Anything else
        runs the interpreter's helper whole, for the atomicity emit_load
        explains, and that is also the path where a store landing on a page
        of translated code gets its blocks dropped */
    bool emit_store(const IRInsn& in, size_t idx, uint32_t& free_mask,
                    X64Gpr arg0, X64Gpr arg1, X64Gpr arg2)
    {
        (void)arg2;
        const X64Gpr rea  = X64Gpr(this->reg_of[in.a]);
        const X64Gpr rval = X64Gpr(this->reg_of[in.b]);
        const bool ea_dies  = this->last_use[in.a] == idx;
        const bool val_dies = this->last_use[in.b] == idx;

        uint32_t avail = free_mask & ~bit(uint8_t(rea)) & ~bit(uint8_t(rval));
        if (!avail) {
            return false;
        }
        const X64Gpr rtmp = X64Gpr(lowest_bit(avail));
        avail &= ~bit(uint8_t(rtmp));
        if (!avail) {
            return false;
        }
        const X64Gpr rtag = X64Gpr(lowest_bit(avail));

        X64Emitter::Label slow = this->asmb.new_label();
        X64Emitter::Label done = this->asmb.new_label();

        this->asmb.mov_reg_reg32(rtmp, rea);
        this->asmb.shr_reg_imm8(rtmp, PPC_PAGE_SIZE_BITS);
        this->asmb.and_reg_imm32(rtmp, TLB_SIZE - 1);
        this->asmb.shl_reg_imm8(rtmp, TLB_ENTRY_SHIFT);
        this->asmb.mov_reg_imm64(REG_SCRATCH, uint64_t(uintptr_t(&pCurDTLB1)));
        this->asmb.add_reg64_mem(rtmp, REG_SCRATCH, 0);

        this->asmb.mov_reg_reg32(rtag, rea);
        this->asmb.and_reg_imm32(rtag, PPC_PAGE_MASK);
        this->asmb.or_reg_mem32(rtag, REG_TIME, this->depoch_disp);
        this->asmb.cmp_mem_reg32(rtmp, TLB_TAG_OFFSET, rtag);
        this->asmb.jcc(X64Cond::NotEqual, slow);

        // both flags, so an and against the pair and a compare, not a test
        this->asmb.movzx_reg16_mem(rtag, rtmp, TLB_FLAGS_OFFSET);
        this->asmb.and_reg_imm32(rtag, TLB_STORE_READY);
        this->asmb.cmp_reg_imm32(rtag, TLB_STORE_READY);
        this->asmb.jcc(X64Cond::NotEqual, slow);

        if (in.width > 1) {
            this->asmb.test_reg_imm32(rea, in.width - 1);
            this->asmb.jcc(X64Cond::NotEqual, slow);
        }

        this->asmb.mov_reg64_mem(rtmp, rtmp, TLB_HOSTW_OFFSET);
        this->asmb.add_reg64_reg64(rtmp, rea);

        switch (in.width) {
        case 1:
            this->asmb.mov_mem_reg8(rtmp, 0, rval);
            break;
        case 2:
            // sthbrx wants the low half exactly as the register holds it,
            // which on this host is already the mirror of guest order
            if (in.byte_reverse) {
                this->asmb.mov_mem_reg16(rtmp, 0, rval);
                break;
            }
            // the value must not be disturbed, it may be live past this
            this->asmb.mov_reg_reg32(rtag, rval);
            this->asmb.bswap_reg32(rtag);
            this->asmb.shr_reg_imm8(rtag, 16);
            this->asmb.mov_mem_reg16(rtmp, 0, rtag);
            break;
        default:
            if (in.byte_reverse) {
                this->asmb.mov_mem_reg32(rtmp, 0, rval);
                break;
            }
            this->asmb.mov_reg_reg32(rtag, rval);
            this->asmb.bswap_reg32(rtag);
            this->asmb.mov_mem_reg32(rtmp, 0, rtag);
            break;
        }

        // rA writeback of the u forms, after the access as the helper does it
        if (in.ureg != IR_NO_UPDATE) {
            this->asmb.mov_mem_reg32(REG_STATE, gpr_offset(in.ureg), rea);
        }
        this->asmb.jmp(done);

        this->asmb.bind(slow);
        if (!this->emit_memory_slow_call(in, rtmp, arg0, arg1)) {
            return false;
        }

        this->asmb.bind(done);

        if (ea_dies)  free_mask |= bit(uint8_t(rea));
        if (val_dies) free_mask |= bit(uint8_t(rval));
        return true;
    }

    /** A guest branch, every form: b and bc carry the target, bclr and bcctr
        read it from LR or CTR the moment they run.

        Everything BO selects is decided here rather than emitted: whether the
        CTR is decremented at all, whether it is tested, and against which
        polarity. A plain b comes through with BO 20 and collapses to the two
        stores at the bottom.

        No IR value is live at this point. The branch takes no operands and
        every guest register write went straight to memory, so the pool is
        free to use as scratch */
    void emit_branch(const IRInsn& in, uint32_t insn_count) {
        const int32_t guest_off = int32_t(in.offset);

        // a run time target has to be read before anything below can change
        // it: blrl overwrites LR with the return address, and the target is
        // the LR it came in with
        if (in.target != BranchTarget::Direct) {
            this->asmb.mov_reg_mem32(RDX, REG_STATE,
                in.target == BranchTarget::LR ? LR_OFFSET : CTR_OFFSET);
        }

        // LR is written whether or not the branch is taken
        if (in.link) {
            this->asmb.lea_reg_mem(RAX, REG_ENTRYPC, guest_off + 4);
            this->asmb.mov_mem_reg32(REG_STATE, LR_OFFSET, RAX);
        }

        // the 750 leaves the counter alone on bcctr, see ppc_bcctr; the test
        // below then reads the copy from before the branch, which is the same
        const bool spares_ctr = in.target == BranchTarget::CTR;
        if (!(in.bo & 0x04) && !spares_ctr) {
            this->asmb.dec_mem32(REG_STATE, CTR_OFFSET);
        }

        X64Emitter::Label not_taken = this->asmb.new_label();

        // ctr_ok: BO2 set skips the test; otherwise BO3 picks whether the
        // branch wants the counter at zero or away from it
        if (!(in.bo & 0x04)) {
            this->live_cmp.valid = false; // the counter test owns FLAGS now
            if (spares_ctr) {
                this->asmb.cmp_reg_imm32(RDX, 0);
            } else {
                this->asmb.cmp_mem_imm8(REG_STATE, CTR_OFFSET, 0);
            }
            this->asmb.jcc((in.bo & 0x02) ? X64Cond::NotEqual : X64Cond::Equal,
                           not_taken);
        }

        // cnd_ok: BO0 set skips the test; otherwise BO1 picks whether the
        // branch wants the CR bit set or clear. When the FLAGS of the
        // compare that wrote this very field are still live, the branch
        // jumps straight off them; the fall through side keeps them live,
        // so a second branch on the same compare fuses too
        if (!(in.bo & 0x10)) {
            const uint8_t cr_bit = in.bi & 3;
            if (this->live_cmp.valid && this->live_cmp.crf == (in.bi & 0x1C) &&
                cr_bit != 3) {
                X64Cond set_c, clr_c;
                switch (cr_bit) {
                case 0: // LT
                    set_c = this->live_cmp.sig ? X64Cond::Less : X64Cond::Below;
                    clr_c = this->live_cmp.sig ? X64Cond::GreaterEqual
                                               : X64Cond::AboveEqual;
                    break;
                case 1: // GT
                    set_c = this->live_cmp.sig ? X64Cond::Greater : X64Cond::Above;
                    clr_c = this->live_cmp.sig ? X64Cond::LessEqual
                                               : X64Cond::BelowEqual;
                    break;
                default: // EQ
                    set_c = X64Cond::Equal;
                    clr_c = X64Cond::NotEqual;
                    break;
                }
                this->asmb.jcc((in.bo & 0x08) ? clr_c : set_c, not_taken);
            } else {
                // the reload test scratches the FLAGS a compare left behind,
                // so a later branch on the compare's own field must not ride
                // them anymore. The shape that bites: compare into one field,
                // a dot form writing cr0, a branch on the first field falling
                // back to this test, then a branch on cr0
                this->live_cmp.valid = false;
                this->asmb.test_mem_imm32(REG_STATE, CR_OFFSET, 0x80000000UL >> in.bi);
                this->asmb.jcc((in.bo & 0x08) ? X64Cond::Equal : X64Cond::NotEqual,
                               not_taken);
            }
        }

        // taken. The interpreter reports this by raising EXEF_BRANCH and
        // leaving the target in ppc_next_instruction_address, for a caller
        // that then copies it into the PC. A block knows the target itself
        // and is already writing the PC on every other way out, so it writes
        // it here too and raises nothing. What the loop outside sees is the
        // same PC either way, one global and one store cheaper
        if (in.target != BranchTarget::Direct) {
            // returns and computed jumps: the target sits in RDX, and the
            // address predicted chain catches the common case of a site that
            // keeps going back to the same place
            this->asmb.and_reg_imm32(RDX, ~3u);
            if (!this->emit_va_chained_exit(insn_count)) {
                this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, RDX);
                this->asmb.mov_reg_imm32(REG_RETIRED, insn_count);
                this->asmb.jmp_abs(this->dispatch);
            }
        } else if (in.absolute) {
            // an absolute target is a fixed virtual address, which is
            // exactly what the address predicted chain guards
            this->asmb.mov_reg_imm32(RDX, in.imm);
            if (!this->emit_va_chained_exit(insn_count)) {
                this->asmb.mov_mem_reg32(REG_STATE, PC_OFFSET, RDX);
                this->asmb.mov_reg_imm32(REG_RETIRED, insn_count);
                this->asmb.jmp_abs(this->dispatch);
            }
        } else {
            // the taken side of every loop lands here, which is what makes
            // this the chain that pays: a tight loop becomes native end to
            // end. A target outside the page falls back to the guarded kind
            const int32_t next_off = guest_off + int32_t(in.imm);
            if (!this->emit_chained_exit(next_off, insn_count)) {
                this->asmb.lea_reg_mem(RDX, REG_ENTRYPC, next_off);
                if (!this->emit_va_chained_exit(insn_count)) {
                    this->emit_exit(next_off, insn_count);
                }
            }
        }

        this->asmb.bind(not_taken);
    }

    /** One per Call, reached only when that Call raised or unwound.

        The return value is still in the low byte of the result register: 1
        when the instruction completed and 0 when it unwound, which is exactly
        how much it adds to the count of what retired before it */
    /** One per cycle settling point, reached when servicing a timer raised
        something. Everything owed went into the counter before the check, so
        there is nothing left to hand to the dispatcher */
    void emit_sync_exits() {
        for (X64Emitter::Label leave : this->sync_exits) {
            this->asmb.bind(leave);
            this->asmb.xor_reg_reg32(REG_RETIRED, REG_RETIRED);
            this->asmb.jmp_abs(this->dispatch);
        }
    }

    bool measure_time_globals() {
        const int64_t icyc  = disp_from_state(&g_icycles);
        const int64_t dline = disp_from_state(&g_icycles_max);
        const int64_t timer = disp_from_state(const_cast<const bool*>(&exec_timer));
        const int64_t gen   = disp_from_state(&mmu_itrans_generation);
        const int64_t depo  = disp_from_state(&g_dtlb_epoch);
        const int64_t iepo  = disp_from_state(&g_itlb_epoch);

        if (icyc  < INT32_MIN || icyc  > INT32_MAX ||
            dline < INT32_MIN || dline > INT32_MAX ||
            timer < INT32_MIN || timer > INT32_MAX ||
            gen   < INT32_MIN || gen   > INT32_MAX ||
            depo  < INT32_MIN || depo  > INT32_MAX ||
            iepo  < INT32_MIN || iepo  > INT32_MAX) {
            return false;
        }

        this->icycles_disp  = int32_t(icyc);
        this->deadline_disp = int32_t(dline);
        this->timer_disp    = int32_t(timer);
        this->gen_disp      = int32_t(gen);
        this->depoch_disp   = int32_t(depo);
        this->iepoch_disp   = int32_t(iepo);
        return true;
    }

    void emit_cold_exits(X64Gpr ret) {
        for (const ColdExit& c : this->cold_exits) {
            this->asmb.bind(c.label);
            this->asmb.movzx_reg8(REG_SCRATCH, ret);
            this->asmb.mov_reg_imm32(REG_RETIRED, c.guest_idx);
            this->asmb.add_reg_reg32(REG_RETIRED, REG_SCRATCH);
            // the PC was written where the raise could happen, so unlike the
            // ordinary exits there is nothing left to say about it
            this->asmb.jmp_abs(this->dispatch);
        }
    }

    /** Calls something whose address is not already in a pinned register.
        Goes through a scratch register because the target can be anywhere in
        the address space and a direct call only reaches two gigabytes */
    void call_absolute(uint64_t target) {
        this->asmb.mov_reg_imm64(REG_SCRATCH, target);
        this->asmb.call_reg(REG_SCRATCH);
    }

    /** The compare whose FLAGS are still in the host's, left by the last
        fusible emit_set_cr: while nothing since scratched them, a branch
        testing that CR field jumps straight off the compare */
    struct {
        bool    valid;
        bool    sig;
        uint8_t crf;
    } live_cmp = {};

    const AbiDesc&        abi;
    const int32_t         pad;
    const uint32_t        pool;
    CodeMem               code;

    /** The shared stubs, emitted once by init and never reclaimed. Blocks
        reach the dispatch one by a rel32 fixed up after they are copied */
    BlockEntry            trampoline = nullptr;
    const void*           dispatch   = nullptr;

    bool                  full = false; // the pool ran out, see wants_flush

    int32_t               icycles_disp  = 0;
    int32_t               deadline_disp = 0;
    int32_t               timer_disp    = 0;
    int32_t               gen_disp      = 0;
    int32_t               depoch_disp   = 0;
    int32_t               iepoch_disp   = 0;

    /** Guest instructions of the block already handed to the retired counter
        by a settling point, so every exit knows what is still owed */
    uint32_t              accounted = 0;

    std::vector<X64Emitter::Label> sync_exits;

    X64Emitter            asmb;
    std::vector<uint32_t> last_use;
    std::vector<uint8_t>  reg_of;
    std::vector<uint8_t>  immediate_const;
    std::vector<uint32_t> immediate_value;
    std::vector<uint8_t>  dead_gpr_store;

    typedef struct ColdExit { X64Emitter::Label label; uint32_t guest_idx; } ColdExit;
    std::vector<ColdExit> cold_exits;

    /** One per chained exit: the slot it jumps through and where its
        resolver thunk begins, as an offset the final code address turns into
        the slot's initial content */
    typedef struct ChainExit { ChainSlot* slot; size_t thunk_off; } ChainExit;
    std::vector<ChainExit> chain_exits;

    /** The address predicted kind, for targets only known as a virtual
        address. The label is how the guards in the exit reach a thunk that
        is only emitted after the body */
    typedef struct VaChainExit {
        ChainVaSlot*      slot;
        size_t            thunk_off;
        X64Emitter::Label thunk;
    } VaChainExit;
    std::vector<VaChainExit> va_chain_exits;

    /** Virtual address of the block being emitted, for the same page test.
        Diagnostics aside, nothing else emitted may depend on it */
    uint32_t cur_virt_addr = 0;
};

} // namespace

const AbiDesc& abi_for_host() {
#if defined(_WIN32)
    return abi_win64;
#else
    return abi_sysv;
#endif
}

std::unique_ptr<Backend> make_x86_64_backend() {
    X86_64Backend* backend = new X86_64Backend(abi_for_host());
    if (!backend->init()) {
        LOG_F(WARNING, "JIT: no code memory, leaving execution on the interpreter");
        delete backend;
        return nullptr;
    }
    return std::unique_ptr<Backend>(backend);
}

} // namespace dppc_jit
