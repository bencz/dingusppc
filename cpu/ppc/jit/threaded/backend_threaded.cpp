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

/** @file Portable backend: executes the IR directly instead of emitting.

    It is not faster than the interpreter and is not meant to be. It exists so
    that a WebAssembly build has a backend at all, so an emitter always has
    somewhere to send the blocks it declines, and above all so that every rule
    the IR encodes has a second implementation to be checked against. The
    tests run the same guest code through this and through the emitter and
    require the results to agree instruction for instruction.

    Its structure is the shape a native backend has to reproduce, which is
    why it is worth reading before the x86-64 one: where the PC is written,
    where retirement is counted, and which of those are allowed to be skipped
    because the operation cannot fail.
 */

#include "../backend.h"
#include "../jitruntime.h"
#include "../../ppcemu.h"

#include <loguru.hpp>

#include <vector>

namespace dppc_jit {

namespace {

typedef struct ThreadedPayload {
    std::vector<IRInsn> insns;
    uint32_t            insn_count;

    /** One slot per IR instruction, since SSA writes each exactly once.

        It lives with the block instead of being built per run: allocating it
        on every execution turned out to cost more than everything the IR
        saved. The emulator runs one guest thread, and a block is never
        re-entered while it is running, so sharing it is safe */
    mutable std::vector<uint32_t> vals;
} ThreadedPayload;

inline uint32_t rot_mask(unsigned mb, unsigned me) {
    uint32_t begin = 0xFFFFFFFFUL >> mb;
    uint32_t end   = me >= 31 ? 0 : 0xFFFFFFFFUL >> (me + 1);
    uint32_t mask  = begin ^ end;
    return (me < mb) ? ~mask : mask;
}

inline uint32_t rotl32(uint32_t v, unsigned sh) {
    return sh ? ((v << sh) | (v >> (32 - sh))) : v;
}

/** ppc_setsoov, word for word: SO and OV set together, OV cleared alone.
    The caller hands in the same operands the interpreter hands in, which for
    the add family means ~b rather than b */
inline void set_soov(uint32_t a, uint32_t b, uint32_t d) {
    if (int32_t((a ^ b) & (a ^ d)) < 0) {
        ppc_state.spr[SPR::XER] |= XER::SO | XER::OV;
    } else {
        ppc_state.spr[SPR::XER] &= ~XER::OV;
    }
}

void threaded_entry(const JitBlock* blk) {
    const ThreadedPayload* payload = static_cast<const ThreadedPayload*>(blk->payload);

    // where the block was entered through, which is not necessarily the
    // virtual address it was translated through: the cache is keyed by
    // physical address and a page can be mapped more than once
    const uint32_t entry_pc = ppc_state.pc;

    std::vector<uint32_t>& vals = payload->vals;

    uint32_t accounted = 0;   // guest instructions already handed to the timer
    uint32_t retired   = payload->insn_count; // unless a Call cuts it short

    for (size_t i = 0; i < payload->insns.size(); i++) {
        const IRInsn& in = payload->insns[i];

        switch (in.opcode) {
        case IROpcode::LoadGPR:
            vals[i] = ppc_state.gpr[in.reg];
            break;
        case IROpcode::StoreGPR:
            ppc_state.gpr[in.reg] = vals[in.a];
            break;
        case IROpcode::LoadSPR:
            vals[i] = ppc_state.spr[in.reg];
            break;
        case IROpcode::StoreSPR:
            ppc_state.spr[in.reg] = vals[in.a];
            break;
        case IROpcode::PcRel:
            vals[i] = entry_pc + in.imm;
            break;

        case IROpcode::ConstI32:
            vals[i] = in.imm;
            break;
        case IROpcode::Add: {
            const uint32_t a = vals[in.a];
            const uint32_t b = vals[in.b];
            const uint32_t d = a + b;
            if (in.oe) set_soov(a, ~b, d);
            vals[i] = d;
            break;
        }
        case IROpcode::Sub: {
            const uint32_t a = vals[in.a];
            const uint32_t b = vals[in.b];
            const uint32_t d = a - b;
            if (in.oe) set_soov(a, b, d);
            vals[i] = d;
            break;
        }

        // the XER[CA] family replicates the interpreter helpers word for
        // word, quirks included; jitir.h says which formula belongs to whom
        case IROpcode::AddCA: {
            const uint32_t a = vals[in.a];
            const uint32_t b = vals[in.b];
            const uint32_t d = a + b;
            if (d < a) ppc_state.spr[SPR::XER] |=  XER::CA;
            else       ppc_state.spr[SPR::XER] &= ~XER::CA;
            if (in.oe) set_soov(a, ~b, d);
            vals[i] = d;
            break;
        }
        case IROpcode::AddECA: {
            const uint32_t a  = vals[in.a];
            const uint32_t b  = vals[in.b];
            const uint32_t ca = !!(ppc_state.spr[SPR::XER] & XER::CA);
            const uint32_t d  = a + b + ca;
            if ((d < a) || (ca && d == a)) ppc_state.spr[SPR::XER] |=  XER::CA;
            else                           ppc_state.spr[SPR::XER] &= ~XER::CA;
            if (in.oe) set_soov(a, ~b, d);
            vals[i] = d;
            break;
        }
        case IROpcode::SubCA: {
            const uint32_t a = vals[in.a];
            const uint32_t b = vals[in.b];
            const uint32_t d = a - b;
            if (a >= b) ppc_state.spr[SPR::XER] |=  XER::CA;
            else        ppc_state.spr[SPR::XER] &= ~XER::CA;
            if (in.oe) set_soov(a, b, d);
            vals[i] = d;
            break;
        }
        case IROpcode::SubECA: {
            const uint32_t a  = vals[in.a];
            const uint32_t bv = vals[in.b];
            const uint32_t ca = !!(ppc_state.spr[SPR::XER] & XER::CA);
            const uint32_t d  = ~a + bv + ca;
            if ((ca && bv == 0xFFFFFFFFUL) || (d < ~a))
                ppc_state.spr[SPR::XER] |=  XER::CA;
            else
                ppc_state.spr[SPR::XER] &= ~XER::CA;
            if (in.oe) set_soov(bv, a, d);
            vals[i] = d;
            break;
        }

        case IROpcode::MulLow: {
            const int64_t product =
                int64_t(int32_t(vals[in.a])) * int64_t(int32_t(vals[in.b]));
            if (in.oe) {
                if (product != int64_t(int32_t(product))) {
                    ppc_state.spr[SPR::XER] |= XER::SO | XER::OV;
                } else {
                    ppc_state.spr[SPR::XER] &= ~XER::OV;
                }
            }
            vals[i] = uint32_t(product);
            break;
        }
        case IROpcode::MulHighS:
            vals[i] = uint32_t((int64_t(int32_t(vals[in.a])) *
                                int64_t(int32_t(vals[in.b]))) >> 32);
            break;
        case IROpcode::MulHighU:
            vals[i] = uint32_t((uint64_t(vals[in.a]) * uint64_t(vals[in.b])) >> 32);
            break;

        case IROpcode::MtCrf:
            ppc_state.cr = (ppc_state.cr & ~in.imm) | (vals[in.a] & in.imm);
            break;
        case IROpcode::And:
            vals[i] = vals[in.a] & vals[in.b];
            break;
        case IROpcode::Or:
            vals[i] = vals[in.a] | vals[in.b];
            break;
        case IROpcode::Xor:
            vals[i] = vals[in.a] ^ vals[in.b];
            break;
        case IROpcode::RotlMask:
            vals[i] = rotl32(vals[in.a], in.sh) & rot_mask(in.mb, in.me);
            break;
        case IROpcode::Exts:
            vals[i] = in.width == 1 ? uint32_t(int32_t(int8_t(vals[in.a])))
                                    : uint32_t(int32_t(int16_t(vals[in.a])));
            break;

        /* Loads and stores run the interpreter's own helper for the whole
           instruction, exactly as the emitted slow path does: effective
           address, access, rd and any update writeback in one piece. What a
           block adds over the interpreter is grouping, and grouping may not
           split an instruction; jitir.h tells the story of the device read
           that was retried because rd was still unwritten */
        case IROpcode::Load:
            ppc_state.pc = entry_pc + in.offset;
            if (!rt_call_op(in.helper, in.imm)) {
                retired = in.insn_idx;
                goto done;
            }
            vals[i] = ppc_state.gpr[in.reg];
            if (exec_flags) {
                retired = in.insn_idx + 1;
                goto done;
            }
            break;

        case IROpcode::Store:
            ppc_state.pc = entry_pc + in.offset;
            if (!rt_call_op(in.helper, in.imm)) {
                retired = in.insn_idx;
                goto done;
            }
            if (exec_flags) {
                retired = in.insn_idx + 1;
                goto done;
            }
            break;

        case IROpcode::SetCR: {
            // same shape as ppc_cmpi and ppc_changecrf0: one of the three
            // ordering bits, then XER[SO] copied down into the field
            const uint32_t x = vals[in.a];
            const uint32_t y = vals[in.b];
            uint32_t cmp_c;
            if (in.cr_signed) {
                cmp_c = (int32_t(x) == int32_t(y)) ? 0x20000000UL
                      : (int32_t(x) >  int32_t(y)) ? 0x40000000UL : 0x80000000UL;
            } else {
                cmp_c = (x == y) ? 0x20000000UL : (x > y) ? 0x40000000UL : 0x80000000UL;
            }
            const uint32_t xercon = (ppc_state.spr[SPR::XER] & XER::SO) >> 3;
            ppc_state.cr = (ppc_state.cr & ~(0xF0000000UL >> in.crf))
                         | ((cmp_c + xercon) >> in.crf);
            break;
        }

        case IROpcode::Branch: {
            // same order the interpreter uses in ppc_bc and ppc_bclr: the CTR
            // decrement happens before the test, the target is read before
            // the link write, and LR is written whether or not the branch is
            // taken. bcctr neither decrements nor tests a decremented value,
            // see ppc_bcctr, whose 750 path leaves the counter alone
            const uint32_t guest_pc = entry_pc + in.offset;

            if (!(in.bo & 0x04) && in.target != BranchTarget::CTR) {
                ppc_state.spr[SPR::CTR]--;
            }
            const uint32_t ctr_ok = (in.bo & 0x04) |
                ((ppc_state.spr[SPR::CTR] != 0) == !(in.bo & 0x02));
            const uint32_t cnd_ok = (in.bo & 0x10) |
                (!(ppc_state.cr & (0x80000000UL >> in.bi)) == !(in.bo & 0x08));

            if (ctr_ok && cnd_ok) {
                switch (in.target) {
                case BranchTarget::Direct:
                    ppc_next_instruction_address =
                        in.absolute ? in.imm : guest_pc + in.imm;
                    break;
                case BranchTarget::LR:
                    ppc_next_instruction_address =
                        ppc_state.spr[SPR::LR] & ~3UL;
                    break;
                case BranchTarget::CTR:
                    ppc_next_instruction_address =
                        ppc_state.spr[SPR::CTR] & ~3UL;
                    break;
                }
                exec_flags = EXEF_BRANCH;
            }
            if (in.link) {
                ppc_state.spr[SPR::LR] = guest_pc + 4;
            }

            // a side exit ends the block only when taken; the block ending
            // kind always does, with everything retired either way
            if (in.flags & IR_BRANCH_SIDE_EXIT) {
                if (!(ctr_ok && cnd_ok)) {
                    break;
                }
                retired = in.insn_idx + 1;
            }
            goto done;
        }

        case IROpcode::Call: {
            const uint32_t guest_idx = in.insn_idx;

            // helpers read the PC for branch targets and exception
            // bookkeeping, and so does the accounting below: an asynchronous
            // exception delivered while settling the count has to resume on
            // this instruction, which has not run yet
            ppc_state.pc = entry_pc + in.offset;

            if (in.flags & IR_SYNC_CYCLES) [[unlikely]] {
                if (!rt_sync_cycles(guest_idx - accounted)) {
                    retired = guest_idx;
                    accounted = guest_idx;
                    goto done;
                }
                accounted = guest_idx;
            }

            if (!rt_call_op(in.helper, in.imm)) {
                // unwound out of the MMU or a device, so it never retired
                retired = guest_idx;
                goto done;
            }
            if (exec_flags) {
                retired = guest_idx + 1;
                goto done;
            }
            break;
        }

        default:
            // Phi has no producer yet and nothing else should exist
            ABORT_F("JIT: threaded backend met an IR opcode it does not run");
        }
    }

done:
    rt_block_end(entry_pc, blk->byte_size, retired - accounted);
}

class ThreadedBackend : public Backend {
public:
    const char* name() const override {
        return "threaded";
    }

    JitBlock* compile(const IRBlock& ir) override {
        ThreadedPayload* payload = new ThreadedPayload;
        payload->insns      = ir.insns;
        payload->insn_count = ir.insn_count;
        payload->vals.resize(ir.insns.size());

        JitBlock* blk   = new JitBlock;
        blk->virt_addr  = ir.virt_addr;
        blk->phys_addr  = ir.phys_addr;
        blk->mode       = ir.mode;
        blk->byte_size  = ir.byte_size;
        blk->insn_count = ir.insn_count;
        blk->end_reason = uint8_t(ir.end_reason);
        blk->entry      = threaded_entry;
        blk->payload    = payload;
        blk->heat       = 0;
        blk->code       = nullptr; // ordinary C++, there is no frame to share
        blk->owner      = nullptr; // the facade fills this in
        blk->chain_in     = nullptr;
        blk->chained_prev = nullptr;
        blk->chained_next = nullptr;

        return blk;
    }

    void release(JitBlock* blk) override {
        delete static_cast<ThreadedPayload*>(blk->payload);
        delete blk;
    }

    void release_all() override {
        // every block was handed out through the code cache, which releases
        // them one at a time, and there is no pool behind them to sweep
    }
};

} // namespace

std::unique_ptr<Backend> make_threaded_backend() {
    return std::unique_ptr<Backend>(new ThreadedBackend);
}

} // namespace dppc_jit
