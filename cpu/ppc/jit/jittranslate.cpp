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

/** @file PowerPC to IR. Runs once per block and serves every backend.

    Only a subset is decoded into real operations; everything else becomes a
    Call and behaves exactly as it did before. The subset was picked from a
    dynamic profile of a Mac OS X 10.0 install boot, 6.1 billion instructions,
    and covers the arithmetic that carries the most weight without touching
    the condition register. CR needs lazy flags to be worth emitting, and
    rushing it here would have locked in the eager version.
 */

#include "jitir.h"
#include "../ppcemu.h"
#include "../ppcmmu.h"

namespace dppc_jit {

namespace {

inline uint32_t primary_op(uint32_t opcode) {
    return opcode >> 26;
}

/** Extended opcode, bits 21 to 30 */
inline uint32_t ext_op(uint32_t opcode) {
    return (opcode >> 1) & 0x3FF;
}

inline bool rc_bit(uint32_t opcode) {
    return opcode & 1;
}

/** SPR number as mfspr and mtspr decode it, low and high halves swapped */
inline uint32_t spr_num(uint32_t opcode) {
    return (((opcode >> 11) & 0x1F) << 5) | ((opcode >> 16) & 0x1F);
}

bool spr_reads_time(uint32_t spr) {
    switch (spr) {
    case SPR::RTCU_U: case SPR::RTCL_U: case SPR::DEC_U:
    case SPR::RTCU_S: case SPR::RTCL_S: case SPR::DEC_S:
    case SPR::TBL_U:  case SPR::TBU_U:
    case SPR::TBL_S:  case SPR::TBU_S:
        return true;
    default:
        return false;
    }
}

bool spr_changes_translation(uint32_t spr) {
    if (spr >= 528 && spr <= 543) { // IBAT0U through DBAT3L
        return true;
    }
    switch (spr) {
    case SPR::SDR1:
    case SPR::HID0: // also flips endian on the 601
    case SPR::HID1:
        return true;
    default:
        return false;
    }
}

bool is_branch(uint32_t opcode) {
    switch (primary_op(opcode)) {
    case 16: // bc and friends
    case 18: // b and friends
        return true;
    case 19:
        return ext_op(opcode) == 16 ||  // bclr
               ext_op(opcode) == 528;   // bcctr
    default:
        return false;
    }
}

/** An always taken direct branch the translator can walk through instead of
    ending the block: its whole effect is the optional LR write and moving
    the decode cursor. Measured before built: branches with a static target
    close 79% of the emitted code time of a Cheetah boot storm, and every
    walk through removes one block transition, one dispatch and one chain
    slot from that path.

    Three fences, each load bearing. Relative only: the block cache is keyed
    by physical address, so a block entered through another mapping of the
    page still has every relative target at the same offset, while an
    absolute target could name different memory entirely. Same page only:
    the host pointer and the invalidation key stop at the page edge. Forward
    only: covered bytes stay inside [virt_addr, virt_addr + byte_size), so
    the invalidation range stays one honest interval, and the instruction
    budget alone bounds the walk, self loops included.

    The LK forms still write LR, which makes bcl 20,31,$+4, the idiom
    position independent code reads its own address with, dissolve into a
    plain constant store */
bool follow_target(uint32_t op, uint32_t pc, uint32_t entry, uint32_t* target) {
    uint32_t t;
    switch (primary_op(op)) {
    case 18: { // b and bl, relative
        if (op & 2) {
            return false;
        }
        int32_t li = int32_t((op & ~3UL) << 6) >> 6;
        t = pc + uint32_t(li);
        break;
    }
    case 16: { // bc always and bcl always, relative: no CR test, no CTR
        if (((op >> 21) & 0x14) != 0x14 || (op & 2)) {
            return false;
        }
        t = pc + uint32_t(int32_t(int16_t(op & ~3UL)));
        break;
    }
    default:
        return false;
    }
    if (((t ^ entry) & PPC_PAGE_MASK) != 0 || t <= pc) {
        return false;
    }
    *target = t;
    return true;
}

bool ends_context(uint32_t opcode) {
    switch (primary_op(opcode)) {
    case 17: // sc
        return true;
    case 19:
        return ext_op(opcode) == 50 ||   // rfi
               ext_op(opcode) == 150;    // isync
    case 31:
        switch (ext_op(opcode)) {
        case 146: // mtmsr
        case 210: // mtsr
        case 242: // mtsrin
            return true;
        case 467: // mtspr
            return spr_changes_translation(spr_num(opcode));
        default:
            return false;
        }
    default:
        return false;
    }
}

bool reads_virtual_time(uint32_t opcode) {
    if (primary_op(opcode) != 31) {
        return false;
    }
    switch (ext_op(opcode)) {
    case 371: // mftb
        return true;
    case 339: // mfspr
    case 467: // mtspr
        return spr_reads_time(spr_num(opcode));
    default:
        return false;
    }
}

/** Mask running from bit mb to bit me inclusive, wrapping when mb > me.
    Same rule the interpreter uses, see rot_mask in ppcopcodes.cpp */
inline uint32_t rot_mask(unsigned mb, unsigned me) {
    uint32_t begin = 0xFFFFFFFFUL >> mb;
    uint32_t end   = me >= 31 ? 0 : 0xFFFFFFFFUL >> (me + 1);
    uint32_t mask  = begin ^ end;
    return (me < mb) ? ~mask : mask;
}

/** Decodes into an IRBlock while keeping guest registers in values.

    A guest register read hits the cache, so a register touched several times
    in a row is loaded once instead of once per reader. Writes go straight
    out; see store_gpr for why deferring them lost more than it gained at this
    block length. In SSA both fall out of numbering the values, with no pass */
class Builder {
public:
    Builder(IRBlock& block) : out(block) {
        for (int i = 0; i < 32; i++) {
            this->cache[i] = IR_NO_VALUE;
        }
    }

    IRValue load_gpr(unsigned reg) {
        if (this->cache[reg] != IR_NO_VALUE) {
            return this->cache[reg];
        }
        IRInsn in = blank(IROpcode::LoadGPR);
        in.reg = uint8_t(reg);
        IRValue v = this->out.append(in);
        this->cache[reg] = v;
        return v;
    }

    /** r0 reads as a literal zero in the addi and addis forms, which is not
        the same as reading GPR 0 */
    IRValue load_gpr_or_zero(unsigned reg) {
        return reg == 0 ? this->constant(0) : this->load_gpr(reg);
    }

    /** The store goes out right away and the value stays cached for reads.

        Holding writes back until the end of the block was the first shape of
        this and it does not pay: a block averages 4.3 guest instructions, so
        a register is seldom written twice, while every deferred write keeps a
        value alive to the block exit. Fourteen registers written meant
        fourteen values live at once against eight host registers, and the
        emitter had to decline. Deferral becomes worth its pressure when a
        compilation unit spans several basic blocks; the IR already allows it */
    void store_gpr(unsigned reg, IRValue v) {
        IRInsn in = blank(IROpcode::StoreGPR);
        in.reg = uint8_t(reg);
        in.a   = v;
        this->out.append(in);
        this->cache[reg] = v;
    }

    IRValue constant(uint32_t imm) {
        IRInsn in = blank(IROpcode::ConstI32);
        in.imm = imm;
        return this->out.append(in);
    }

    /** Only for SPRs that are plain storage; the decoder guards which */
    IRValue load_spr(unsigned spr) {
        IRInsn in = blank(IROpcode::LoadSPR);
        in.reg = uint8_t(spr);
        return this->out.append(in);
    }

    void store_spr(unsigned spr, IRValue v) {
        IRInsn in = blank(IROpcode::StoreSPR);
        in.reg = uint8_t(spr);
        in.a   = v;
        this->out.append(in);
    }

    IRValue binary(IROpcode op, IRValue a, IRValue b, bool oe = false) {
        IRInsn in = blank(op);
        in.a  = a;
        in.b  = b;
        in.oe = oe;
        return this->out.append(in);
    }

    void mtcrf(IRValue a, uint32_t mask) {
        IRInsn in = blank(IROpcode::MtCrf);
        in.a   = a;
        in.imm = mask;
        this->out.append(in);
    }

    IRValue rotl_mask(IRValue a, unsigned sh, unsigned mb, unsigned me) {
        IRInsn in = blank(IROpcode::RotlMask);
        in.a  = a;
        in.sh = uint8_t(sh);
        in.mb = uint8_t(mb);
        in.me = uint8_t(me);
        return this->out.append(in);
    }

    IRValue sign_extend(IRValue a, unsigned width) {
        IRInsn in = blank(IROpcode::Exts);
        in.a     = a;
        in.width = uint8_t(width);
        return this->out.append(in);
    }

    /** A helper can change any register, so nothing cached survives it */
    void invalidate() {
        for (int i = 0; i < 32; i++) {
            this->cache[i] = IR_NO_VALUE;
        }
    }

    IRValue load(IRValue ea, unsigned width, bool is_signed, uint32_t raw,
                 PPCOpcode helper, unsigned rd, unsigned update_reg,
                 bool byte_reverse) {
        IRInsn in       = blank(IROpcode::Load);
        in.a            = ea;
        in.width        = uint8_t(width);
        in.signed_load  = is_signed;
        in.byte_reverse = byte_reverse;
        in.imm          = raw;    // the slow path hands this to the helper
        in.helper       = helper; // which does the whole instruction at once
        in.reg          = uint8_t(rd);
        in.ureg         = uint8_t(update_reg);
        IRValue v = this->out.append(in);

        // the slow path is a call, so nothing cached may outlive it. Keeping
        // the read cache across a load would need the live values spilled
        // around that call, and a load is common enough that the simple rule
        // wins: reads after a load come from memory again
        this->invalidate();
        return v;
    }

    /** crf is already four times the field number, the way the instruction
        encodes it. Materialised right here; see the note on IROpcode::SetCR
        for why there is nothing to defer at this block length */
    void set_cr(unsigned crf, IRValue a, IRValue b, bool is_signed) {
        IRInsn in     = blank(IROpcode::SetCR);
        in.crf        = uint8_t(crf);
        in.a          = a;
        in.b          = b;
        in.cr_signed  = is_signed;
        this->out.append(in);
    }

    /** The Rc bit, which is a signed comparison of the result against zero */
    void set_cr0(IRValue result) {
        this->set_cr(0, result, this->constant(0), true);
    }

    void store(IRValue ea, IRValue value, unsigned width, uint32_t raw,
               PPCOpcode helper, unsigned update_reg, bool byte_reverse) {
        IRInsn in       = blank(IROpcode::Store);
        in.a            = ea;
        in.b            = value;
        in.width        = uint8_t(width);
        in.byte_reverse = byte_reverse;
        in.imm          = raw;
        in.helper       = helper;
        in.ureg         = uint8_t(update_reg);
        this->out.append(in);
        this->invalidate(); // the slow path is a call, same as for a load
    }

    /** imm is the target when absolute, otherwise the displacement from the
        address of this very instruction. The LR and CTR targets ignore both */
    void branch(uint8_t bo, uint8_t bi, bool link, bool absolute, uint32_t imm,
                BranchTarget target = BranchTarget::Direct) {
        IRInsn in    = blank(IROpcode::Branch);
        in.bo        = bo;
        in.bi        = bi;
        in.link      = link;
        in.absolute  = absolute;
        in.imm       = imm;
        in.target    = target;
        this->out.append(in);
    }

    void call(PPCOpcode helper, uint32_t raw, uint8_t flags) {
        IRInsn in  = blank(IROpcode::Call);
        in.helper  = helper;
        in.imm     = raw;
        in.flags  |= flags;
        this->out.append(in);
        this->invalidate();
    }

    void set_position(uint32_t off, uint32_t idx) {
        this->offset   = uint16_t(off);
        this->insn_idx = uint16_t(idx);
    }

    IRBlock& block() { return this->out; }

private:
    IRInsn blank(IROpcode op) {
        IRInsn in{};
        in.opcode   = op;
        in.flags    = 0;
        in.offset   = this->offset;
        in.insn_idx = this->insn_idx;
        in.a        = IR_NO_VALUE;
        in.b        = IR_NO_VALUE;
        in.dest     = IR_NO_VALUE;
        in.type     = IRType::I32;
        in.ureg     = IR_NO_UPDATE;
        in.helper   = nullptr;
        return in;
    }

    IRBlock& out;
    IRValue  cache[32];
    uint16_t offset   = 0;
    uint16_t insn_idx = 0;
};

/** The D and X form loads. rA of zero reads as a literal zero, the same rule
    addi follows, and the update forms hand rA to the Load itself, whose fast
    path writes the effective address back and whose slow path leaves it to
    the helper. Splitting the update into IR of its own would redo it after a
    slow path that already did it.

    lwzu with rA zero or rA equal to rD is an invalid form; the interpreter
    handles it and this leaves it there rather than guessing */
bool decode_load(Builder& b, uint32_t op, PPCOpcode helper) {
    unsigned width;
    bool     is_signed = false;
    bool     update    = false;
    bool     reversed  = false;
    bool     x_form    = false;

    switch (primary_op(op)) {
    case 32: width = 4; break;                        // lwz
    case 33: width = 4; update = true; break;         // lwzu
    case 34: width = 1; break;                        // lbz
    case 35: width = 1; update = true; break;         // lbzu
    case 40: width = 2; break;                        // lhz
    case 41: width = 2; update = true; break;         // lhzu
    case 42: width = 2; is_signed = true; break;      // lha
    case 43: width = 2; is_signed = true; update = true; break; // lhau
    case 31:
        x_form = true;
        switch (ext_op(op)) {
        case 23:  width = 4; break;                       // lwzx
        case 55:  width = 4; update = true; break;        // lwzux
        case 87:  width = 1; break;                       // lbzx
        case 119: width = 1; update = true; break;        // lbzux
        case 279: width = 2; break;                       // lhzx
        case 311: width = 2; update = true; break;        // lhzux
        case 343: width = 2; is_signed = true; break;     // lhax
        case 375: width = 2; is_signed = true; update = true; break; // lhaux
        case 534: width = 4; reversed = true; break;      // lwbrx
        case 790: width = 2; reversed = true; break;      // lhbrx
        default:  return false;
        }
        break;
    default: return false;
    }

    const unsigned rd = (op >> 21) & 31;
    const unsigned ra = (op >> 16) & 31;
    const unsigned rb = (op >> 11) & 31;
    const int32_t  d  = int32_t(int16_t(op));

    if (update && (ra == 0 || ra == rd)) {
        return false; // invalid form, the interpreter owns it
    }

    IRValue ea = x_form
        ? b.binary(IROpcode::Add, b.load_gpr_or_zero(ra), b.load_gpr(rb))
        : b.binary(IROpcode::Add, b.load_gpr_or_zero(ra), b.constant(uint32_t(d)));
    b.store_gpr(rd, b.load(ea, width, is_signed, op, helper, rd,
                           update ? ra : IR_NO_UPDATE, reversed));
    return true;
}

/** The compare forms, which write a condition register field and nothing
    else, plus andi. and andis., which write a register and CR0 */
bool decode_compare(Builder& b, uint32_t op) {
    const unsigned crf = (op >> 21) & 0x1C;
    const unsigned ra  = (op >> 16) & 31;
    const unsigned rb  = (op >> 11) & 31;
    const unsigned rs  = (op >> 21) & 31;

    switch (primary_op(op)) {
    case 11: // cmpi
        b.set_cr(crf, b.load_gpr(ra), b.constant(uint32_t(int32_t(int16_t(op)))), true);
        return true;
    case 10: // cmpli
        b.set_cr(crf, b.load_gpr(ra), b.constant(uint16_t(op)), false);
        return true;
    case 28: { // andi.
        IRValue r = b.binary(IROpcode::And, b.load_gpr(rs), b.constant(uint16_t(op)));
        b.store_gpr(ra, r);
        b.set_cr0(r);
        return true;
    }
    case 29: { // andis.
        IRValue r = b.binary(IROpcode::And, b.load_gpr(rs),
                             b.constant(uint32_t(uint16_t(op)) << 16));
        b.store_gpr(ra, r);
        b.set_cr0(r);
        return true;
    }
    case 31:
        // the L bit picks 64 bit comparison, which this processor has not got
        if (op & 0x200000) {
            return false;
        }
        switch (ext_op(op)) {
        case 0:  // cmp
            b.set_cr(crf, b.load_gpr(ra), b.load_gpr(rb), true);
            return true;
        case 32: // cmpl
            b.set_cr(crf, b.load_gpr(ra), b.load_gpr(rb), false);
            return true;
        default:
            return false;
        }
    default:
        return false;
    }
}

/** The D and X form stores, mirroring decode_load, update forms included.
    stu with rA of zero is invalid and stays with the interpreter */
bool decode_store(Builder& b, uint32_t op, PPCOpcode helper) {
    unsigned width;
    bool     update   = false;
    bool     reversed = false;
    bool     x_form   = false;

    switch (primary_op(op)) {
    case 36: width = 4; break;                    // stw
    case 37: width = 4; update = true; break;     // stwu
    case 38: width = 1; break;                    // stb
    case 39: width = 1; update = true; break;     // stbu
    case 44: width = 2; break;                    // sth
    case 45: width = 2; update = true; break;     // sthu
    case 31:
        x_form = true;
        switch (ext_op(op)) {
        case 151: width = 4; break;                   // stwx
        case 183: width = 4; update = true; break;    // stwux
        case 215: width = 1; break;                   // stbx
        case 247: width = 1; update = true; break;    // stbux
        case 407: width = 2; break;                   // sthx
        case 439: width = 2; update = true; break;    // sthux
        case 662: width = 4; reversed = true; break;  // stwbrx
        case 918: width = 2; reversed = true; break;  // sthbrx
        default:  return false;
        }
        break;
    default: return false;
    }

    const unsigned rs = (op >> 21) & 31;
    const unsigned ra = (op >> 16) & 31;
    const unsigned rb = (op >> 11) & 31;
    const int32_t  d  = int32_t(int16_t(op));

    if (update && ra == 0) {
        return false;
    }

    IRValue ea = x_form
        ? b.binary(IROpcode::Add, b.load_gpr_or_zero(ra), b.load_gpr(rb))
        : b.binary(IROpcode::Add, b.load_gpr_or_zero(ra), b.constant(uint32_t(d)));
    IRValue val = b.load_gpr(rs);
    b.store(ea, val, width, op, helper, update ? ra : IR_NO_UPDATE, reversed);
    return true;
}

/** All four branch forms. b and bc carry their target in the instruction;
    bclr and bcctr read it from LR or CTR when they run */
bool decode_branch(Builder& b, uint32_t op) {
    switch (primary_op(op)) {
    case 18: { // b, ba, bl, bla
        int32_t li = int32_t((op & ~3UL) << 6) >> 6;
        // BO 20 is "always", which skips both the CTR and the CR test
        b.branch(20, 0, op & 1, (op >> 1) & 1, uint32_t(li));
        return true;
    }
    case 16: { // bc, bca, bcl, bcla
        int32_t bd = int32_t(int16_t(op & ~3UL));
        b.branch((op >> 21) & 0x1F, (op >> 16) & 0x1F,
                 op & 1, (op >> 1) & 1, uint32_t(bd));
        return true;
    }
    case 19:
        switch (ext_op(op)) {
        case 16: // bclr, bclrl
            b.branch((op >> 21) & 0x1F, (op >> 16) & 0x1F,
                     op & 1, false, 0, BranchTarget::LR);
            return true;
        case 528: // bcctr, bcctrl
            // the 601 decrements CTR on the way through and this machine is
            // not one, but the flag is runtime state, so respect it
            if (is_601) {
                return false;
            }
            b.branch((op >> 21) & 0x1F, (op >> 16) & 0x1F,
                     op & 1, false, 0, BranchTarget::CTR);
            return true;
        default:
            return false;
        }
    default:
        return false;
    }
}

/** mfspr and mtspr of the SPRs that are plain storage, plus eieio and mtcrf.

    LR and CTR bracket every function call as mflr and mtlr, and their helper
    does nothing but move a word; anything with privilege, time or the MMU
    behind it stays a Call. eieio orders storage accesses an emulator keeps in
    order anyway, its helper is empty, so nothing is emitted at all and the
    instruction only counts as retired */
bool decode_spr(Builder& b, uint32_t op) {
    if (primary_op(op) != 31 || rc_bit(op)) {
        return false;
    }

    if (ext_op(op) == 854) { // eieio
        return true;
    }

    if (ext_op(op) == 144) { // mtcrf, whose mask CRM decides once, right here
        const unsigned crm = (op >> 12) & 0xFF;
        uint32_t mask = 0;
        for (int i = 0; i < 8; i++) {
            if (crm & (0x80 >> i)) {
                mask |= 0xF0000000UL >> (i * 4);
            }
        }
        b.mtcrf(b.load_gpr((op >> 21) & 31), mask);
        return true;
    }

    const unsigned spr = spr_num(op);
    if (spr != SPR::LR && spr != SPR::CTR) {
        return false;
    }
    const unsigned rd = (op >> 21) & 31;

    switch (ext_op(op)) {
    case 339: // mfspr
        b.store_gpr(rd, b.load_spr(spr));
        return true;
    case 467: // mtspr
        b.store_spr(spr, b.load_gpr(rd));
        return true;
    default:
        return false;
    }
}

/** Tries to decode one guest instruction into real operations.

    Returns false when the instruction is outside the emitted subset, which
    leaves the caller to emit a Call. The Rc forms go through SetCR and the
    OE forms through the oe flag, both picked straight from the boot profile:
    addco. alone was 8% of a Mac OS 9 boot still running as a helper */
bool decode_alu(Builder& b, uint32_t op) {
    const unsigned rd = (op >> 21) & 31;
    const unsigned ra = (op >> 16) & 31;
    const unsigned rb = (op >> 11) & 31;
    const unsigned rs = rd; // same field, named by the form

    switch (primary_op(op)) {
    case 7: // mulli
        b.store_gpr(rd, b.binary(IROpcode::MulLow, b.load_gpr(ra),
                                 b.constant(uint32_t(int32_t(int16_t(op))))));
        return true;

    case 8: // subfic, whose CA rule the SubCA note in jitir.h explains
        b.store_gpr(rd, b.binary(IROpcode::SubCA,
                                 b.constant(uint32_t(int32_t(int16_t(op)))),
                                 b.load_gpr(ra)));
        return true;

    case 12: case 13: { // addic and addic.
        IRValue r = b.binary(IROpcode::AddCA, b.load_gpr(ra),
                             b.constant(uint32_t(int32_t(int16_t(op)))));
        b.store_gpr(rd, r);
        if (primary_op(op) == 13) b.set_cr0(r);
        return true;
    }

    case 14: { // addi
        int32_t simm = int32_t(int16_t(op));
        b.store_gpr(rd, b.binary(IROpcode::Add, b.load_gpr_or_zero(ra),
                                 b.constant(uint32_t(simm))));
        return true;
    }
    case 15: { // addis
        int32_t simm = int32_t(int16_t(op));
        b.store_gpr(rd, b.binary(IROpcode::Add, b.load_gpr_or_zero(ra),
                                 b.constant(uint32_t(simm) << 16)));
        return true;
    }
    case 24: // ori
        b.store_gpr(ra, b.binary(IROpcode::Or, b.load_gpr(rs),
                                 b.constant(uint16_t(op))));
        return true;
    case 25: // oris
        b.store_gpr(ra, b.binary(IROpcode::Or, b.load_gpr(rs),
                                 b.constant(uint32_t(uint16_t(op)) << 16)));
        return true;
    case 26: // xori
        b.store_gpr(ra, b.binary(IROpcode::Xor, b.load_gpr(rs),
                                 b.constant(uint16_t(op))));
        return true;
    case 27: // xoris
        b.store_gpr(ra, b.binary(IROpcode::Xor, b.load_gpr(rs),
                                 b.constant(uint32_t(uint16_t(op)) << 16)));
        return true;

    case 20: { // rlwimi and rlwimi., the only rotate that keeps part of rA
        unsigned sh = (op >> 11) & 31;
        unsigned mb = (op >> 6) & 31;
        unsigned me = (op >> 1) & 31;
        IRValue rot  = b.rotl_mask(b.load_gpr(rs), sh, mb, me);
        IRValue keep = b.binary(IROpcode::And, b.load_gpr(ra),
                                b.constant(~rot_mask(mb, me)));
        IRValue r = b.binary(IROpcode::Or, rot, keep);
        b.store_gpr(ra, r);
        if (rc_bit(op)) b.set_cr0(r);
        return true;
    }

    case 21: { // rlwinm and rlwinm.
        unsigned sh = (op >> 11) & 31;
        unsigned mb = (op >> 6) & 31;
        unsigned me = (op >> 1) & 31;
        IRValue r = b.rotl_mask(b.load_gpr(rs), sh, mb, me);
        b.store_gpr(ra, r);
        if (rc_bit(op)) b.set_cr0(r);
        return true;
    }

    case 31: {
        IRValue r;
        unsigned dest;
        // bit 0x200 of the extended opcode is OE wherever both forms exist,
        // which is why the arithmetic cases below come in pairs
        const bool oe = (ext_op(op) & 0x200) != 0;
        switch (ext_op(op)) {
        case 266: case 778: // add, addo
            r = b.binary(IROpcode::Add, b.load_gpr(ra), b.load_gpr(rb), oe);
            dest = rd;
            break;
        case 40: case 552: // subf, subfo
            r = b.binary(IROpcode::Sub, b.load_gpr(rb), b.load_gpr(ra), oe);
            dest = rd;
            break;
        case 104: case 616: // neg, nego, which is subf from zero
            r = b.binary(IROpcode::Sub, b.constant(0), b.load_gpr(ra), oe);
            dest = rd;
            break;
        case 8: case 520: // subfc, subfco, rB - rA with CA saying no borrow
            r = b.binary(IROpcode::SubCA, b.load_gpr(rb), b.load_gpr(ra), oe);
            dest = rd;
            break;
        case 10: case 522: // addc, addco
            r = b.binary(IROpcode::AddCA, b.load_gpr(ra), b.load_gpr(rb), oe);
            dest = rd;
            break;
        case 138: case 650: // adde, addeo
            r = b.binary(IROpcode::AddECA, b.load_gpr(ra), b.load_gpr(rb), oe);
            dest = rd;
            break;
        case 136: case 648: // subfe, subfeo
            r = b.binary(IROpcode::SubECA, b.load_gpr(ra), b.load_gpr(rb), oe);
            dest = rd;
            break;
        case 202: case 714: // addze, addzeo, which is adde against zero
            r = b.binary(IROpcode::AddECA, b.load_gpr(ra), b.constant(0), oe);
            dest = rd;
            break;
        case 235: case 747: // mullw, mullwo
            r = b.binary(IROpcode::MulLow, b.load_gpr(ra), b.load_gpr(rb), oe);
            dest = rd;
            break;
        case 75: // mulhw
            r = b.binary(IROpcode::MulHighS, b.load_gpr(ra), b.load_gpr(rb));
            dest = rd;
            break;
        case 11: // mulhwu
            r = b.binary(IROpcode::MulHighU, b.load_gpr(ra), b.load_gpr(rb));
            dest = rd;
            break;
        case 444: // or, which is also mr when rs == rb
            r = b.binary(IROpcode::Or, b.load_gpr(rs), b.load_gpr(rb));
            dest = ra;
            break;
        case 28: // and
            r = b.binary(IROpcode::And, b.load_gpr(rs), b.load_gpr(rb));
            dest = ra;
            break;
        case 316: // xor
            r = b.binary(IROpcode::Xor, b.load_gpr(rs), b.load_gpr(rb));
            dest = ra;
            break;
        case 954: // extsb
            r = b.sign_extend(b.load_gpr(rs), 1);
            dest = ra;
            break;
        case 922: // extsh
            r = b.sign_extend(b.load_gpr(rs), 2);
            dest = ra;
            break;
        default:
            return false;
        }
        b.store_gpr(dest, r);
        if (rc_bit(op)) b.set_cr0(r);
        return true;
    }

    default:
        return false;
    }
}

} // namespace

bool translate_block(uint32_t virt_addr, uint32_t phys_addr, const uint8_t* code,
                     uint32_t mode, IRBlock& out)
{
    out.reset(virt_addr, phys_addr, mode);
    Builder b(out);

    // a block never crosses a page, so the host pointer stays valid for the
    // whole walk and ppc_code_cache_add gets a range it can key by one page.
    // The cursor is a byte offset from the entry because a walked through
    // branch moves it by more than 4; reset() preset SizeLimit, so running
    // out of budget needs no store here
    const uint32_t page_limit = PPC_PAGE_SIZE - (virt_addr & ~PPC_PAGE_MASK);
    uint32_t off = 0;

    for (uint32_t i = 0; i < jit_max_block_insns; i++) {
        uint32_t raw = ppc_read_instruction(code + off);

        // resolving the helper through the current table is what makes MSR[FP]
        // part of the block key: the no FPU table maps different functions
        PPCOpcode helper = ppc_opcode_grabber[(raw >> 15 & 0x1F800) | (raw & 0x7FF)];

        if (helper == ppc_illegalop) {
            // stop short of it so the PC the block exits with is the address
            // of the instruction that did not run, which is what a caller
            // waiting on a particular PC expects to see
            out.end_reason = BlockEnd::Untranslatable;
            break;
        }

        b.set_position(off, i);

        // walking through a branch retires it without emitting anything but
        // the LR write, and decoding continues at its target. Gated with the
        // branch group so bisection still sees every branch end its block
        uint32_t follow_to;
        if ((jit_decode_groups & JIT_DECODE_BRANCH) &&
            follow_target(raw, virt_addr + off, virt_addr, &follow_to)) {
            if (raw & 1) {
                b.store_spr(SPR::LR, b.constant(virt_addr + off + 4));
            }
            out.byte_size  = off + 4;
            out.insn_count++;
            out.end_word = raw;
            off = follow_to - virt_addr;
            continue;
        }

        // the groups are switchable so a misbehaviour can be bisected against
        // a real workload: turn one off and whatever it covered goes to a
        // helper instead, which is the interpreter's own code
        const bool decoded =
            (jit_decode_groups & JIT_DECODE_ALU     && decode_alu(b, raw))          ||
            (jit_decode_groups & JIT_DECODE_BRANCH  && decode_branch(b, raw))       ||
            (jit_decode_groups & JIT_DECODE_LOAD    && decode_load(b, raw, helper)) ||
            (jit_decode_groups & JIT_DECODE_STORE   && decode_store(b, raw, helper))||
            (jit_decode_groups & JIT_DECODE_COMPARE && decode_compare(b, raw))      ||
            (jit_decode_groups & JIT_DECODE_SPR     && decode_spr(b, raw));

        if (!decoded) {
            const bool sync = reads_virtual_time(raw) || jit_sync_every_call;
            b.call(helper, raw, sync ? IR_SYNC_CYCLES : 0);
        }

        out.byte_size  = off + 4;
        out.insn_count++;
        out.end_word = raw;

        if (is_branch(raw)) {
            out.end_reason = BlockEnd::Branch;
            break;
        }
        if (ends_context(raw)) {
            out.end_reason = BlockEnd::ContextSync;
            break;
        }

        off += 4;
        if (off >= page_limit) {
            out.end_reason = BlockEnd::PageEnd;
            break;
        }
    }

    return out.insn_count != 0;
}

} // namespace dppc_jit
