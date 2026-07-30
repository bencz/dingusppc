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

/** @file Intermediate representation for a translated block.

    PowerPC is decoded into this once and every backend reads it, so the
    decoding rules and the block boundary rules live in one place instead of
    once per host architecture.

    The form is SSA: an instruction defines at most one value and a value is
    never reassigned. That costs nothing to maintain in straight line code,
    which is all a basic block contains, and it is what makes the guest
    register cache below fall out for free instead of needing a pass.

    There are no PHI nodes and none are needed. A PHI is only required where
    control flow merges, and neither a basic block nor a trace with side exits
    ever merges: one entry, one or more ways out, nothing rejoins. PHI becomes
    necessary the day a whole loop is compiled as one region, and the opcode
    slot is reserved for that. Building the machinery before then would be
    paying for a case that cannot arise.

    Guest state is reached only through LoadGPR and StoreGPR. Making it
    explicit is what lets the translator keep a register in a value across
    several instructions and write it back once, and it is also what makes the
    two places where state must be coherent obvious: before any call that can
    raise, and at every exit.
 */

#ifndef PPC_JIT_IR_H
#define PPC_JIT_IR_H

#include "../ppcemu.h"

#include <cinttypes>
#include <vector>

namespace dppc_jit {

/** Index into IRBlock::insns of the instruction that defined the value.
    Values are SSA, so the defining instruction identifies the value */
typedef uint16_t IRValue;
constexpr IRValue IR_NO_VALUE = 0xFFFF;

/** ureg of a load or store that is not an update form */
constexpr uint8_t IR_NO_UPDATE = 0xFF;

/** f64 exists because floating point will matter once something other than an
    installer is running; the boot profile shows it at 0.1% but that is a
    property of the workload, not of the guest.

    v128 is deliberately absent. The emulated processor is an MPC750, which
    has no AltiVec at all, so no guest code can produce a vector operation
    however it is compiled. It belongs here the day a G4 machine does */
enum class IRType : uint8_t {
    I32,
    F64,
};

enum class IROpcode : uint8_t {
    /** Hand the raw instruction word to an interpreter helper. Everything
        outside the emitted subset looks like this, and the instructions that
        stay in a helper forever keep looking like this */
    Call,

    /** Reserved. See the note on merges at the top of this file */
    Phi,

    // guest state
    LoadGPR,   // reg          -> dest
    StoreGPR,  // reg <- a
    LoadSPR,   // spr `reg`    -> dest. Only for SPRs that are plain storage,
               // which today means LR and CTR; anything with a side effect
               // stays a Call
    StoreSPR,  // spr `reg` <- a

    // values
    ConstI32,  // imm          -> dest

    // integer arithmetic and logic
    Add,       // a + b        -> dest
    Sub,       // a - b        -> dest
    And,       // a & b        -> dest
    Or,        // a | b        -> dest
    Xor,       // a ^ b        -> dest
    RotlMask,  // rotl(a, sh) & rot_mask(mb, me) -> dest
    Exts,      // sign extend a from `width` bytes -> dest

    /** The XER[CA] family. Each replicates the formula of the interpreter
        helper it displaces, not the architecture book, because the tests hold
        the three implementations to bit equality and the interpreter is the
        one being matched.

        AddCA   a + b            CA = carry out
        AddECA  a + b + CA       CA = carry out, addze is b == 0
        SubCA   a - b            CA = no borrow, covers subfc and subfic
        SubECA  ~a + b + CA      CA = carry out, which is subfe */
    AddCA,
    AddECA,
    SubCA,
    SubECA,

    /** The multiply family. MulLow is the low word of the signed product,
        which is also the low word of the unsigned one, so it serves mulli and
        mullw both; the high word forms differ and get an opcode each */
    MulLow,
    MulHighS,
    MulHighU,

    /** mtcrf: merges `a` into the condition register under a mask decided at
        translation time from CRM, carried in imm. No value is defined */
    MtCrf,

    /** A guest branch. The b and bc forms carry their target in the
        instruction; bclr and bcctr take theirs from LR or CTR at run time,
        which `target` distinguishes.

        It always ends the block. Taken means writing the target to
        ppc_next_instruction_address and raising EXEF_BRANCH, exactly as the
        interpreter does, so the execution loop cannot tell the two apart.
        Not taken means falling out of the block and letting the PC advance */
    Branch,

    /** A guest load. `a` is the effective address, already computed, and
        `width` says how many bytes; `signed_load` picks lha over lhz. `reg`
        is the destination GPR and `ureg` the update register of the u forms,
        0xFF when there is none; `helper` is the interpreter's own routine
        for the instruction.

        The emitter inlines the primary TLB hit on an aligned access: value
        into the defined SSA value, and for the u forms the effective address
        into `ureg`, which is why the address is a value here rather than a
        base and a displacement. Everything else runs the helper whole. The
        instruction has to stay atomic: a device read can raise an exception
        and still complete, and an exit taken between the access and the
        register writes leaves rd unwritten with SRR0 pointing back at the
        instruction, so the guest reruns it and the device sees the read
        twice. That is not a theory, it cost a day: the interpreter never
        splits an instruction, so neither may a block.

        The defined value always equals what the instruction leaves in `reg`,
        and the translator only ever feeds it to that StoreGPR. The slow path
        relies on this: after the helper has done everything, it reloads the
        value from the register file instead of redoing the transforms */
    Load,

    /** A guest store. `a` is the effective address and `b` the value, with
        `reg`, `ureg` and `helper` as for Load.

        The inline fast path is stricter than the load one: besides the tag
        and the alignment it demands PAGE_WRITABLE and PTE_SET_C, because a
        page that is merely readable, or one holding translated code, or one
        whose PTE change bit has not been set yet, all need work the emitted
        path does not do. The slow path is the helper, whole, for the same
        atomicity reason Load explains */
    Store,

    /** Writes one condition register field from a comparison of `a` against
        `b`, signed or not, plus the summary overflow bit copied out of XER.

        The Rc forms come through here too, as a signed comparison of the
        result against zero, which is what ppc_changecrf0 computes.

        It is materialised where it appears rather than deferred. Deferring is
        what "lazy flags" means and the slot for it is the translator's
        pending state, but at a mean block length of 4.3 guest instructions
        the dominant shape is a compare immediately consumed by the branch
        that ends the block, so there is no dead write to drop. It pays once a
        compilation unit spans several blocks */
    SetCR,
};

enum IRFlags : uint8_t {
    /** The instruction reads virtual time, which is derived from g_icycles,
        so retired instructions have to be accounted for before it runs and
        not merely when the block ends */
    IR_SYNC_CYCLES = 1 << 0,
};

/** Where a Branch finds its target */
enum class BranchTarget : uint8_t {
    Direct, // in the instruction: displacement, or absolute when AA
    LR,     // link register at run time, masked to a word boundary
    CTR,    // count register at run time, likewise
};

typedef struct IRInsn {
    IROpcode opcode;
    uint8_t  flags;

    /** Bytes from the start of the block, not an address.

        Blocks are keyed by physical address, so the same block can be entered
        through more than one virtual mapping of the page it lives on. Nothing
        a backend keeps may be an absolute virtual address for that reason; the
        one the block was translated through is in IRBlock::virt_addr and is
        good for diagnostics only */
    uint16_t offset;

    IRValue  a, b;    // operands, IR_NO_VALUE when unused
    IRValue  dest;    // value defined, IR_NO_VALUE when none
    IRType   type;

    uint32_t imm;     // ConstI32 immediate, or the raw word for Call
    uint8_t  reg;     // GPR for LoadGPR, StoreGPR and Load, SPR for LoadSPR
    uint8_t  ureg;    // update register of the u form loads and stores, 0xFF none
    uint8_t  sh, mb, me;  // RotlMask fields
    uint8_t  width;   // Exts and Load width in bytes
    bool     signed_load;  // Load sign extends rather than zero extends
    bool     byte_reverse; // Load and Store move the bytes mirrored, lwbrx kin
    bool     cr_signed;   // SetCR compares signed rather than unsigned
    uint8_t  crf;         // SetCR field, already multiplied by four

    /** The OE forms: XER[SO|OV] from the signed overflow of the operation,
        set together and OV cleared alone, the way ppc_setsoov does it. Valid
        on Add, Sub, the XER[CA] family and MulLow */
    bool     oe;

    // Branch. bo and bi come straight from the instruction, and everything
    // they select is decided at translation time rather than emitted
    uint8_t  bo, bi;
    bool     link;        // LK, writes the return address to LR
    bool     absolute;    // AA, imm is the target rather than a displacement
    BranchTarget target;  // where the branch finds where it goes

    PPCOpcode helper; // Call
} IRInsn;

/** Why the translator stopped adding instructions to a block */
enum class BlockEnd : uint8_t {
    Branch,         // a branch closed the block, the next PC is a run time value
    ContextSync,    // rfi, sc, mtmsr and friends: the mode may be different next
    PageEnd,        // ran into the end of the guest page
    Untranslatable, // the next instruction has no translation, stop before it
    SizeLimit,      // hit the block length cap
};

const char* block_end_name(BlockEnd reason);

/** One block of guest code, decoded. Never crosses a guest page, which is
    what lets ppc_code_cache_add key it by a single page */
typedef struct IRBlock {
    uint32_t virt_addr;
    uint32_t phys_addr;
    uint32_t mode;
    uint32_t byte_size;

    /** Guest instructions covered, which is no longer insns.size(): one guest
        instruction can decode into several IR instructions, and a guest
        register write can decode into none at all until it is flushed */
    uint32_t insn_count;

    BlockEnd end_reason;
    std::vector<IRInsn> insns;

    void reset(uint32_t virt, uint32_t phys, uint32_t translation_mode);
    IRValue append(const IRInsn& insn);
} IRBlock;

/** Longest run of guest instructions the translator will decode. Nothing
    measured picked this number, it just bounds the worst case latency of a
    single translation.

    Overridable through DPPC_JIT_BLOCK because it is the sharpest tool there
    is for telling two kinds of bug apart: set it to 1 and every block is one
    guest instruction, which makes the block executor observe the state at
    exactly the points the interpreter does. A misbehaviour that survives that
    is in how an instruction is translated, and one that does not is in what
    happens at a block boundary */
constexpr uint32_t JIT_MAX_BLOCK_INSNS = 64;

extern uint32_t jit_max_block_insns;

/** Which groups of instructions the translator turns into real operations
    rather than into a call to the interpreter's own helper.

    All on normally. DPPC_JIT_OPS turns them off one at a time, which is how a
    misbehaviour that only a real workload provokes gets narrowed down to the
    group that causes it without having to reason about which instruction it
    might be */
enum JitDecodeGroup : uint32_t {
    JIT_DECODE_ALU     = 1 << 0,
    JIT_DECODE_BRANCH  = 1 << 1,
    JIT_DECODE_LOAD    = 1 << 2,
    JIT_DECODE_STORE   = 1 << 3,
    JIT_DECODE_COMPARE = 1 << 4,
    JIT_DECODE_SPR     = 1 << 5, // mfspr and mtspr of LR and CTR, eieio, mtcrf
    JIT_DECODE_ALL     = 0x3F,
};

extern uint32_t jit_decode_groups;

/** Settles the retired count before every call into a helper rather than only
    before the ones that read virtual time.

    A diagnostic knob. With it on, and every decode group off, a block observes
    the emulator at exactly the points the interpreter does, while still being
    a block. What that separates is a translation grouped wrongly from a
    translation observed too rarely */
extern bool jit_sync_every_call;

/** Decodes guest code into `out`.

    `code` points at the first instruction in host memory and `phys_addr` is
    its physical address; both come from mmu_translate_imem, which the caller
    already had to run to find out whether the block was cached. The block
    stays inside the page `code` belongs to, so walking the host pointer
    forward is safe for its whole length.

    Returns false when nothing could be decoded, which means the instruction
    at virt_addr has to go back to the interpreter */
bool translate_block(uint32_t virt_addr, uint32_t phys_addr, const uint8_t* code,
                     uint32_t mode, IRBlock& out);

} // namespace dppc_jit

#endif // PPC_JIT_IR_H
