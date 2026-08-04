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

/** @file Tests for the block translator.

    The point of most of these is that running a piece of guest code through
    blocks has to land in exactly the same place as running it one instruction
    at a time, down to the retired instruction count. Everything else here is
    about the two things a block adds that an interpreter never had to worry
    about: when a translation stops being valid, and where a block is allowed
    to end.

    These run last because they call ppc_cpu_init, which resets the processor.
 */

#include "../jit/jitir.h"
#include "../ppccodecache.h"
#include "../ppcemu.h"
#include "../ppcjit.h"
#include "../ppcmmu.h"
#if defined(DPPC_JIT_X86_64)
#include "../jit/x86_64/emitter.h"
#endif
#include "devices/common/mmiodevice.h"
#include "devices/memctrl/mpc106.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

static int jit_tested;
static int jit_failed;

static void jit_check(bool passed, const char* what) {
    jit_tested++;
    if (!passed) {
        cout << "  Failed: " << what << endl;
        jit_failed++;
    }
}

#if defined(DPPC_JIT_X86_64)
/** Exercise exact encodings whose prefix, REX or special zero form is easy to
    get subtly wrong, rather than relying only on the current CPU to accept
    them. */
static void test_x64_encoding() {
    dppc_jit::X64Emitter emitter;
    emitter.movbe_reg_mem32(dppc_jit::R9, dppc_jit::R10, 0x7F);
    emitter.movbe_mem_reg32(dppc_jit::R10, -4, dppc_jit::R9);
    emitter.movbe_mem_reg16(dppc_jit::R10, -8, dppc_jit::R9);
    emitter.cmp_reg_imm32(dppc_jit::R9, 0);
    emitter.cmp_reg_imm32(dppc_jit::R10, 0x12345678);
    emitter.imul_reg_reg_imm32(dppc_jit::R9, dppc_jit::R10, uint32_t(-2));
    emitter.imul_reg_reg_imm32(dppc_jit::R10, dppc_jit::R9, 0x12345678);
    emitter.add_reg_imm32_flags(dppc_jit::R9, 0);
    emitter.adc_reg_imm32(dppc_jit::R10, uint32_t(-1));

    constexpr uint8_t expected[] = {
        0x45, 0x0F, 0x38, 0xF0, 0x4A, 0x7F,
        0x45, 0x0F, 0x38, 0xF1, 0x4A, 0xFC,
        0x66, 0x45, 0x0F, 0x38, 0xF1, 0x4A, 0xF8,
        0x45, 0x85, 0xC9,
        0x41, 0x81, 0xFA, 0x78, 0x56, 0x34, 0x12,
        0x45, 0x6B, 0xCA, 0xFE,
        0x45, 0x69, 0xD1, 0x78, 0x56, 0x34, 0x12,
        0x41, 0x83, 0xC1, 0x00,
        0x41, 0x83, 0xD2, 0xFF,
    };
    bool same = emitter.size() == sizeof(expected);
    for (size_t i = 0; same && i < sizeof(expected); i++) {
        same = emitter.bytes()[i] == expected[i];
    }
    jit_check(same, "MOVBE, compare, IMUL and carry have exact x64 encodings");
}
#endif

/** A device access is part of the guest instruction stream, so it must see
    every instruction retired before it even when the block executor normally
    batches cycle accounting until the exit. */
class JitCycleMmio final : public MMIODevice {
public:
    JitCycleMmio() {
        this->set_name("JIT cycle test");
    }

    uint32_t read(uint32_t, uint32_t, int) override {
        this->read_cycle = g_icycles;
        return uint32_t(g_icycles);
    }

    void write(uint32_t, uint32_t, uint32_t value, int) override {
        this->write_cycle = g_icycles;
        this->write_value = value;
    }

    void reset() {
        this->read_cycle  = UINT64_MAX;
        this->write_cycle = UINT64_MAX;
        this->write_value = 0;
    }

    uint64_t read_cycle  = UINT64_MAX;
    uint64_t write_cycle = UINT64_MAX;
    uint32_t write_value = 0;
};

/*  li     r3, 0
    li     r4, 10
    mtctr  r4
 loop:
    addi   r3, r3, 5
    bdnz   loop
    .long  0            <- illegal, never runs, marks where to stop

    Three blocks worth of shapes in six words: a straight run that ends on a
    branch, a two instruction loop body that gets reused ten times, and a
    fall through into an instruction that has no translation */
static const uint32_t test_code[] = {
    0x38600000, 0x3880000A, 0x7C8903A6, 0x38630005, 0x4200FFFC, 0x00000000
};

constexpr uint32_t CODE_BASE  = 0x1000;
constexpr uint32_t CODE_END   = CODE_BASE + 0x14; // the illegal word
constexpr uint32_t LOOP_START = CODE_BASE + 0x0C;

/** mtctr, the third instruction, which sits inside the first block rather
    than at either end of it */
constexpr uint32_t MID_BLOCK  = CODE_BASE + 0x08;

// expected results, they are the same either way round
constexpr uint32_t EXPECTED_R3      = 50;
constexpr uint64_t EXPECTED_RETIRED = 23; // 3 setup, then 10 times two

typedef struct RunResult {
    uint32_t r3;
    uint32_t ctr;
    uint32_t pc;
    uint64_t retired;
} RunResult;

static void load_test_code() {
    for (size_t i = 0; i < sizeof(test_code) / sizeof(test_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, CODE_BASE + uint32_t(i) * 4, test_code[i]);
    }
}

static RunResult run_test_code(uint32_t goal = CODE_END) {
    ppc_state.pc      = CODE_BASE;
    ppc_state.gpr[3]  = 0xDEADBEEF;
    ppc_state.gpr[4]  = 0;
    ppc_state.spr[SPR::CTR] = 0;
    g_icycles         = 0;
    g_icycles_max     = 0;
    exec_flags        = 0;
    power_on          = true;

    ppc_exec_until(goal);

    return {ppc_state.gpr[3], ppc_state.spr[SPR::CTR], ppc_state.pc, g_icycles};
}

static bool same_run(const RunResult& a, const RunResult& b) {
    return a.r3 == b.r3 && a.ctr == b.ctr && a.pc == b.pc && a.retired == b.retired;
}

constexpr uint32_t COLD_PAGE_BASE = 0x1FF8;
constexpr uint32_t COLD_PAGE_END  = 0x2004;

/** Three addi instructions with the page seam between the second and third.
    A cold span may carry its host pointer to 0x1FFC, but must return for an
    honest fetch before executing 0x2000. */
static RunResult run_cold_page_code() {
    constexpr uint32_t ADDI_R3 = 0x38630001;
    for (uint32_t addr = COLD_PAGE_BASE; addr < COLD_PAGE_END; addr += 4) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, addr, ADDI_R3);
    }
    mmu_write_vmem<uint32_t>(NO_OPCODE, COLD_PAGE_END, 0);

    ppc_state.pc            = COLD_PAGE_BASE;
    ppc_state.gpr[3]        = 0;
    ppc_state.spr[SPR::CTR] = 0;
    g_icycles               = 0;
    g_icycles_max           = 0;
    exec_flags              = 0;
    power_on                = true;

    ppc_exec_until(COLD_PAGE_END);
    return {ppc_state.gpr[3], ppc_state.spr[SPR::CTR], ppc_state.pc, g_icycles};
}

static void set_test_heat_value(const char* value) {
#if defined(_WIN32)
    _putenv_s("DPPC_JIT_HEAT", value);
#else
    setenv("DPPC_JIT_HEAT", value, 1);
#endif
}

static void set_test_heat(bool gated) {
    set_test_heat_value(gated ? "3" : "0");
}

/** Interpreter and JIT have to agree on everything observable */
static void test_parity(const RunResult& interp) {
    jit_check(interp.r3 == EXPECTED_R3, "the interpreter run produced the expected sum");
    jit_check(interp.pc == CODE_END, "the interpreter run stopped at the goal address");
    jit_check(interp.retired == EXPECTED_RETIRED,
              "the interpreter retired the expected instruction count");

    RunResult jitted = run_test_code();

    jit_check(same_run(interp, jitted), "a jitted run lands exactly where an interpreted one does");

    // the goal address holds an instruction with no translation. Reaching it
    // means the translator stopped short of it instead of folding it into the
    // block and taking a program exception
    jit_check(jitted.pc == CODE_END, "a block ends before an untranslatable instruction");

    // one block for the straight run into the branch, one for the loop body
    jit_check(ppc_jit_num_blocks() == 2, "the code translated into two blocks");

    RunResult again = run_test_code();
    jit_check(same_run(interp, again), "a second run off the cached blocks agrees too");
    jit_check(ppc_jit_num_blocks() == 2, "the second run reused the blocks it had");
}

/** icbi is the primary invalidation path, so a block has to go away when the
    guest retires one over it and come back on the next run */
static void test_icbi_invalidation(const RunResult& interp) {
    // icbi 0,r3
    constexpr uint32_t OPCODE_ICBI_R3 = 0x7C001FAC;

    jit_check(ppc_jit_num_blocks() == 2, "starting from two cached blocks");

    ppc_state.gpr[3] = LOOP_START;
    ppc_main_opcode(ppc_opcode_grabber, OPCODE_ICBI_R3);

    // both blocks cover the line the loop body sits on
    jit_check(ppc_jit_num_blocks() == 0, "icbi dropped the blocks over its line");

    RunResult after = run_test_code();
    jit_check(same_run(interp, after), "the retranslated blocks behave the same");
    jit_check(ppc_jit_num_blocks() == 2, "and the code translated again");
}

/** A store into translated code takes the block down through the PAGE_CODE
    path, without the guest having to run icbi at all; a store elsewhere on
    the page leaves the blocks alone and only pays the slow path */
static void test_store_invalidation(const RunResult& interp) {
    jit_check(ppc_jit_num_blocks() == 2, "starting from two cached blocks");

    // somewhere on the code page but past the code itself
    mmu_write_vmem<uint32_t>(NO_OPCODE, CODE_BASE + 0x800, 0x12345678);
    jit_check(ppc_jit_num_blocks() == 2, "a store beside the code drops nothing");

    // straight into the first instruction
    mmu_write_vmem<uint32_t>(NO_OPCODE, CODE_BASE, 0x60000000);
    jit_check(ppc_jit_num_blocks() < 2, "a store into the code dropped its block");

    // put the original instruction back so the reference run still holds
    mmu_write_vmem<uint32_t>(NO_OPCODE, CODE_BASE, test_code[0]);

    RunResult after = run_test_code();
    jit_check(same_run(interp, after), "the code still runs the same afterwards");
}

/** A translation is only valid for the mode it was made under. MSR[FP] picks
    which opcode table the helpers came from, so the same physical address
    under a different MSR[FP] is a different block, not a cache hit */
static void test_mode_is_part_of_the_key(const RunResult& interp) {
    ppc_jit_flush();
    jit_check(ppc_jit_num_blocks() == 0, "starting from an empty cache");

    const uint32_t saved_msr = ppc_state.msr;

    run_test_code();
    jit_check(ppc_jit_num_blocks() == 2, "two blocks under the current mode");

    ppc_msr_did_change(ppc_state.msr, ppc_state.msr | MSR::FP, false);
    exec_flags = 0;

    RunResult with_fp = run_test_code();
    jit_check(same_run(interp, with_fp), "the same code under MSR[FP] behaves the same");
    jit_check(ppc_jit_num_blocks() == 4,
              "the same addresses under a different mode translated separately");

    ppc_msr_did_change(ppc_state.msr, saved_msr, false);
    exec_flags = 0;
}

/** ppc_exec_until compares the PC after every instruction, so an address in
    the middle of a block is one the interpreter stops on. Running the block
    whole would sail past it */
static void test_mid_block_goal(const RunResult& interp) {
    ppc_jit_flush();

    RunResult jitted = run_test_code(MID_BLOCK);

    jit_check(jitted.pc == MID_BLOCK, "a goal inside a block is still stopped on");
    jit_check(same_run(interp, jitted), "and stopping there matches the interpreter");
}

/** The emitter has to agree with the threaded backend instruction for
    instruction, which is the whole reason the threaded one exists.

    Every guest instruction is still a helper call on both sides, so what this
    really checks is the frame the emitter builds around them: the prologue,
    the pinned registers surviving a call, the stack staying aligned, the
    exception boundary, and the retired count coming back right */
static void test_native_matches_threaded(const RunResult& interp) {
    ppc_jit_disable();

    if (!ppc_jit_enable(JitBackend::threaded)) {
        jit_check(false, "the threaded backend came up");
        return;
    }
    RunResult threaded = run_test_code();
    jit_check(ppc_jit_threaded_compiles() == 2 && ppc_jit_native_compiles() == 0,
              "forcing threaded really used the threaded backend");
    jit_check(same_run(interp, threaded), "the threaded backend matches the interpreter");

    ppc_jit_disable();

    if (!ppc_jit_enable(JitBackend::automatic)) {
        RunResult automatic = run_test_code();
        jit_check(!ppc_jit_is_enabled() && same_run(interp, automatic),
                  "a host without an emitter stays on the interpreter");
        return;
    }
    RunResult automatic = run_test_code();

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping the native comparison)" << endl;
        jit_check(same_run(interp, automatic), "the fallback still matches the interpreter");
        return;
    }

    jit_check(ppc_jit_native_compiles() == 2 && ppc_jit_threaded_compiles() == 0,
              "the emitter took both blocks rather than declining them");
    jit_check(same_run(interp, automatic), "emitted code matches the interpreter");
    jit_check(same_run(threaded, automatic), "emitted code matches the threaded backend");

    // the emitter pins callee saved registers across helper calls, so running
    // the same blocks again out of the cache is a second chance to notice one
    // of them coming back wrong
    RunResult again = run_test_code();
    jit_check(same_run(threaded, again), "and still matches on a cached second run");
    jit_check(ppc_jit_native_compiles() == 2, "the second run recompiled nothing");
}

/** Automatic mode leaves cold code on the ordinary interpreter. The heat
    counter must represent real entries, survive without a placeholder block,
    and start over when the translation cache is flushed. */
static void test_interpreter_heat_gate() {
    ppc_jit_disable();

    // One interpreted instruction per run gives the threshold an exact,
    // observable unit. The reference is collected while the JIT is off.
    constexpr uint32_t ONE_INSN_GOAL = CODE_BASE + 4;
    const RunResult interp_one = run_test_code(ONE_INSN_GOAL);
    const RunResult interp_full = run_test_code();
    const RunResult interp_page = run_cold_page_code();

    jit_check(!dppc_jit::jit_fallback_ends_span(0x38630001) &&
                  dppc_jit::jit_fallback_ends_span(0x42000000) &&
                  dppc_jit::jit_fallback_ends_span(0x7C000124),
              "cold spans classify arithmetic, branch and context boundaries");

    // Keep the complete setup-and-loop program below its threshold. Besides
    // parity, its final bdnz is a conditional branch that falls through: the
    // cold span must still return at that boundary so ppc_exec_until observes
    // CODE_END before the illegal word there executes.
    set_test_heat_value("100");
    if (!ppc_jit_enable(JitBackend::automatic)) {
        const RunResult fallback = run_test_code(ONE_INSN_GOAL);
        jit_check(!ppc_jit_is_enabled() && same_run(interp_one, fallback),
                  "automatic mode without an emitter stays on the interpreter");
        set_test_heat(false);
        return;
    }

    const RunResult cold_span = run_test_code();
    jit_check(same_run(interp_full, cold_span),
              "a bounded cold interpreter span matches the interpreter");
    const RunResult cold_page = run_cold_page_code();
    jit_check(same_run(interp_page, cold_page) && cold_page.r3 == 3 &&
                  cold_page.retired == 3,
              "a cold interpreter span stops safely at a page boundary");

    const uint32_t saved_block_limit = dppc_jit::jit_max_block_insns;
    dppc_jit::jit_max_block_insns = 1;
    const RunResult single_step_span = run_test_code();
    dppc_jit::jit_max_block_insns = saved_block_limit;
    jit_check(same_run(interp_full, single_step_span),
              "a one-instruction cold span preserves exact fallback behaviour");

    jit_check(ppc_jit_num_blocks() == 0 && ppc_jit_native_compiles() == 0 &&
                  ppc_jit_threaded_compiles() == 0,
              "a below-threshold span creates no backend blocks");

    ppc_jit_disable();
    set_test_heat(true);
    jit_check(ppc_jit_enable(JitBackend::automatic),
              "the automatic backend returned for the exact heat test");

    const string backend = ppc_jit_backend_name() ? ppc_jit_backend_name() : "";
    jit_check(backend.find("interpreter") != string::npos,
              "automatic mode identifies the interpreter fallback");
    jit_check(backend.find("threaded") == string::npos,
              "automatic mode does not select the threaded backend");

    RunResult first  = run_test_code(ONE_INSN_GOAL);
    RunResult second = run_test_code(ONE_INSN_GOAL);
    jit_check(same_run(interp_one, first) && same_run(interp_one, second),
              "cold entries execute exactly like the interpreter");
    jit_check(ppc_jit_num_blocks() == 0 && ppc_jit_native_compiles() == 0,
              "two entries stay below a heat threshold of three");
    jit_check(ppc_jit_threaded_compiles() == 0,
              "cold automatic entries create no threaded blocks");

    ppc_jit_flush();
    RunResult after_flush = run_test_code(ONE_INSN_GOAL);
    RunResult second_after_flush = run_test_code(ONE_INSN_GOAL);
    jit_check(same_run(interp_one, after_flush) &&
                  same_run(interp_one, second_after_flush),
              "flushed hotness still falls back correctly");
    jit_check(ppc_jit_num_blocks() == 0 && ppc_jit_native_compiles() == 0,
              "a cache flush resets the heat interval");

    RunResult threshold = run_test_code(ONE_INSN_GOAL);
    jit_check(same_run(interp_one, threshold),
              "the threshold entry remains architecturally identical");
    jit_check(ppc_jit_threaded_compiles() == 0,
              "crossing the automatic heat threshold never uses threaded");

    if (ppc_jit_native_compiles()) {
        jit_check(ppc_jit_native_compiles() == 1 && ppc_jit_num_blocks() == 1,
                  "the third real entry emits exactly one native block");
    } else {
        // The deliberately minimal AArch64 bring-up currently declines every
        // block. Its important contract here is still interpreter fallback.
        jit_check(ppc_jit_num_blocks() == 0,
                  "a host emitter decline leaves no placeholder block");
    }

    ppc_jit_disable();
    set_test_heat(false);
    jit_check(ppc_jit_enable(JitBackend::automatic),
              "the automatic backend returned with the heat gate disabled");
}

/*  One per opcode of the emitted subset, plus forms that exercise register
    allocation: reusing the dying operand (addi r5,r5,-1), both operands the
    same (add r5,r5,r5) and the destination landing on the second operand.

    The last word is illegal on purpose and marks where the block ends.  */
static const uint32_t alu_code[] = {
    0x38601234, // li     r3, 0x1234
    0x3C805678, // addis  r4, r0, 0x5678
    0x7CA32214, // add    r5, r3, r4
    0x60A600FF, // ori    r6, r5, 0x00FF
    0x64C71000, // oris   r7, r6, 0x1000
    0x68A8AAAA, // xori   r8, r5, 0xAAAA
    0x7CA93038, // and    r9, r5, r6
    0x7C6A2378, // or     r10, r3, r4
    0x7CAB3278, // xor    r11, r5, r6
    0x7C6C0774, // extsb  r12, r3
    0x7C8D0734, // extsh  r13, r4
    0x54AE403E, // rlwinm r14, r5, 8, 0, 31
    0x54AF2226, // rlwinm r15, r5, 4, 8, 19
    0x54B3083C, // slwi   r19, r5, 1
    0x54B4F800, // slwi   r20, r5, 31
    0x54B5F87E, // srwi   r21, r5, 1
    0x54B60FFE, // srwi   r22, r5, 31
    0x54B8003E, // rlwinm r24, r5, 0, 0, 31  identity
    0x54B93840, // rlwinm r25, r5, 7, 1, 0   wrapping full mask
    0x1F7A0000, // mulli  r27, r26, 0
    0x1F9A0001, // mulli  r28, r26, 1
    0x1FBA0002, // mulli  r29, r26, 2
    0x1FDAFFFE, // mulli  r30, r26, -2
    0x1FFA7FFF, // mulli  r31, r26, 32767
    0x38A5FFFF, // addi   r5, r5, -1
    0x7CA52A14, // add    r5, r5, r5
    0x7CB02B78, // mr     r16, r5
    0x7E232050, // subf   r17, r3, r4
    0x50B2400E, // rlwimi r18, r5, 8, 0, 7
    0x3A63007F, // addi   r19, r3, 127   signed imm8 upper edge
    0x7E770034, // cntlzw r23, r19        helper must observe the first r19
    0x3A630005, // addi   r19, r3, 5      overwrites r19 only after the helper
    0x3A830080, // addi   r20, r3, 128   needs the full immediate
    0x3AA3FF80, // addi   r21, r3, -128  signed imm8 lower edge
    0x3AC3FF7F, // addi   r22, r3, -129  needs the full immediate
    0x00000000, // illegal on purpose, the block stops short of it
};

constexpr uint32_t ALU_BASE = 0x2000;
constexpr uint32_t ALU_END  = ALU_BASE + uint32_t(sizeof(alu_code) - 4);

typedef struct AluResult {
    uint32_t gpr[32];
    uint32_t pc;
    uint64_t retired;
} AluResult;

static void load_alu_code() {
    for (size_t i = 0; i < sizeof(alu_code) / sizeof(alu_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, ALU_BASE + uint32_t(i) * 4, alu_code[i]);
    }
}

static AluResult run_alu_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xA5A50000u | uint32_t(i); // recognizable pattern
    }
    ppc_state.pc  = ALU_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(ALU_END);

    AluResult r;
    for (int i = 0; i < 32; i++) {
        r.gpr[i] = ppc_state.gpr[i];
    }
    r.pc      = ppc_state.pc;
    r.retired = g_icycles;
    return r;
}

static bool same_alu(const AluResult& a, const AluResult& b, int* first_diff) {
    for (int i = 0; i < 32; i++) {
        if (a.gpr[i] != b.gpr[i]) { *first_diff = i; return false; }
    }
    *first_diff = -1;
    return a.pc == b.pc && a.retired == b.retired;
}

/** The subset the emitter covers may not diverge in any of the 32
    registers. Comparing only the final result would hide an allocation bug
    that ruins a register the program never reads again */
static void test_alu_subset() {
    load_alu_code();

    ppc_jit_disable();
    AluResult interp = run_alu_code();
    jit_check(interp.pc == ALU_END, "the ALU program stopped where it should on the interpreter");
    jit_check(interp.gpr[23] == 19,
              "a helper observed the GPR value stored before it");

    ppc_jit_enable(JitBackend::threaded);
    AluResult threaded = run_alu_code();
    int diff = -1;
    jit_check(same_alu(interp, threaded, &diff),
              diff < 0 ? "the threaded backend agrees with the interpreter"
                       : "the threaded backend diverges in a register");
    if (diff >= 0) {
        cout << "    r" << diff << ": interpreter 0x" << hex << interp.gpr[diff]
             << ", threaded 0x" << threaded.gpr[diff] << dec << endl;
    }

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    AluResult native = run_alu_code();

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping the native comparison)" << endl;
        return;
    }

    diff = -1;
    jit_check(same_alu(interp, native, &diff),
              diff < 0 ? "emitted code agrees with the interpreter"
                       : "emitted code diverges in a register");
    if (diff >= 0) {
        cout << "    r" << diff << ": interpreter 0x" << hex << interp.gpr[diff]
             << ", emitted 0x" << native.gpr[diff] << dec << endl;
    }

    jit_check(native.retired == interp.retired,
              "emitted code counted the same retired instructions");

    // run again off the cached block: catches a pinned register coming
    // back wrong from a helper call
    AluResult again = run_alu_code();
    jit_check(same_alu(native, again, &diff), "a second pass over the cached block agrees");
}

/** Constant-only arithmetic is settled while building the IR. Besides
    guarding the folded value, this makes the intended optimisation visible
    without depending on a particular native emitter or disassembler. */
static void test_constant_fold_ir() {
    constexpr uint32_t words[] = {
        0x3C801234, // addis r4, r0, 0x1234
        0x60845678, // ori   r4, r4, 0x5678
        0x00000000, // illegal: ends the block before it
    };
    alignas(4) uint8_t code[sizeof(words)];
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++) {
        code[i * 4 + 0] = uint8_t(words[i] >> 24);
        code[i * 4 + 1] = uint8_t(words[i] >> 16);
        code[i * 4 + 2] = uint8_t(words[i] >> 8);
        code[i * 4 + 3] = uint8_t(words[i]);
    }

    dppc_jit::IRBlock ir;
    const bool translated = dppc_jit::translate_block(0x1000, 0x1000, code, 0, ir);
    jit_check(translated, "the constant construction translated to IR");

    dppc_jit::IRValue last_r4 = dppc_jit::IR_NO_VALUE;
    bool has_runtime_alu = false;
    for (const dppc_jit::IRInsn& in : ir.insns) {
        if (in.opcode == dppc_jit::IROpcode::StoreGPR && in.reg == 4) {
            last_r4 = in.a;
        }
        has_runtime_alu |= in.opcode == dppc_jit::IROpcode::Add ||
                           in.opcode == dppc_jit::IROpcode::Or;
    }
    const bool final_constant = last_r4 != dppc_jit::IR_NO_VALUE &&
        ir.insns[last_r4].opcode == dppc_jit::IROpcode::ConstI32 &&
        ir.insns[last_r4].imm == 0x12345678;
    jit_check(final_constant, "addis plus ori folded to the exact word");
    jit_check(!has_runtime_alu, "the folded construction has no run-time ALU operation");
}

/** Native memory helpers never rejoin their block, so the successful inline
    path may keep unrelated guest values cached. This sequence reads r3 on
    both sides of a load and a store; one LoadGPR is enough for all four uses. */
static void test_memory_gpr_cache_ir() {
    constexpr uint32_t words[] = {
        0x80830000, // lwz  r4, 0(r3)
        0x38A30004, // addi r5, r3, 4
        0x90830008, // stw  r4, 8(r3)
        0x38C3000C, // addi r6, r3, 12
        0x00000000, // illegal: ends the block before it
    };
    alignas(4) uint8_t code[sizeof(words)];
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++) {
        code[i * 4 + 0] = uint8_t(words[i] >> 24);
        code[i * 4 + 1] = uint8_t(words[i] >> 16);
        code[i * 4 + 2] = uint8_t(words[i] >> 8);
        code[i * 4 + 3] = uint8_t(words[i]);
    }

    dppc_jit::IRBlock ir;
    const bool translated = dppc_jit::translate_block(0x1000, 0x1000, code, 0, ir);
    jit_check(translated, "the memory cache sequence translated to IR");

    unsigned r3_loads = 0;
    bool has_load = false;
    bool has_store = false;
    for (const dppc_jit::IRInsn& in : ir.insns) {
        r3_loads += in.opcode == dppc_jit::IROpcode::LoadGPR && in.reg == 3;
        has_load |= in.opcode == dppc_jit::IROpcode::Load;
        has_store |= in.opcode == dppc_jit::IROpcode::Store;
    }
    jit_check(has_load && has_store && r3_loads == 1,
              "load and store fast paths preserve the cached base GPR");
}

/*  D form loads, including the ones that force the slow path: unaligned
    accesses, a negative displacement and the update forms, whose rA may only
    be written after the access succeeds.

    The first pass misses the TLB and the second hits, so running twice
    covers both sides of the inline check.  */
static const uint32_t load_code[] = {
    0x3D400000, // addis r10, r0, 0
    0x614A3000, // ori   r10, r10, 0x3000
    0x806A0000, // lwz   r3, 0(r10)
    0x888A0004, // lbz   r4, 4(r10)
    0xA0AA0006, // lhz   r5, 6(r10)
    0xA8CA0008, // lha   r6, 8(r10)
    0xA8EA000A, // lha   r7, 10(r10)
    0x810A0001, // lwz   r8, 1(r10)    unaligned
    0xA12A0003, // lhz   r9, 3(r10)    unaligned
    0x856A0010, // lwzu  r11, 16(r10)
    0x8D8A0001, // lbzu  r12, 1(r10)
    0xADAA0003, // lhau  r13, 3(r10)
    0x81CAFFFC, // lwz   r14, -4(r10)
    0x39400000, // li    r10, 0          may not erase state loads observed
    0x00000000, // illegal on purpose, the block stops short of it
};

constexpr uint32_t LOAD_BASE = 0x4000;
constexpr uint32_t LOAD_END  = LOAD_BASE + uint32_t(sizeof(load_code) - 4);
constexpr uint32_t LOAD_DATA = 0x3000;

// chosen to exercise sign extension: 0xFFFF is negative, 0x7FFF is not
static const uint32_t load_data[] = {
    0x12345678, 0xAABBCCDD, 0xFFFF7FFF, 0x11223344, 0xDEADBEEF, 0xCAFEBABE,
};

static void load_load_code() {
    for (size_t i = 0; i < sizeof(load_code) / sizeof(load_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, LOAD_BASE + uint32_t(i) * 4, load_code[i]);
    }
    for (size_t i = 0; i < sizeof(load_data) / sizeof(load_data[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, LOAD_DATA + uint32_t(i) * 4, load_data[i]);
    }
}

static AluResult run_load_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0x5A5A0000u | uint32_t(i);
    }
    ppc_state.pc  = LOAD_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(LOAD_END);

    AluResult r;
    for (int i = 0; i < 32; i++) r.gpr[i] = ppc_state.gpr[i];
    r.pc      = ppc_state.pc;
    r.retired = g_icycles;
    return r;
}

/** The inline TLB fast path has to produce what the interpreter does, both
    when it hits and when it drops to the slow path */
static void test_load_subset() {
    load_load_code();

    ppc_jit_disable();
    AluResult interp = run_load_code();
    jit_check(interp.pc == LOAD_END, "the load program stopped where it should");

    ppc_jit_enable(JitBackend::threaded);
    AluResult threaded = run_load_code();
    int diff = -1;
    jit_check(same_alu(interp, threaded, &diff),
              "the threaded backend agrees with the interpreter on loads");
    if (diff >= 0) {
        cout << "    r" << diff << ": interpreter 0x" << hex << interp.gpr[diff]
             << ", threaded 0x" << threaded.gpr[diff] << dec << endl;
    }

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);

    // the first pass misses the TLB, the second hits
    AluResult miss = run_load_code();
    AluResult hit  = run_load_code();

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping the native comparison)" << endl;
        return;
    }

    diff = -1;
    jit_check(same_alu(interp, miss, &diff), "emitted loads agree on the first pass");
    if (diff >= 0) {
        cout << "    r" << diff << ": interpreter 0x" << hex << interp.gpr[diff]
             << ", emitted 0x" << miss.gpr[diff] << dec << endl;
    }
    diff = -1;
    jit_check(same_alu(interp, hit, &diff), "and agree with the TLB already warm");
    if (diff >= 0) {
        cout << "    r" << diff << ": interpreter 0x" << hex << interp.gpr[diff]
             << ", emitted 0x" << hit.gpr[diff] << dec << endl;
    }
}

/*  D form stores, including unaligned ones and the update forms, whose rA
    may only be written after the access succeeds.  */
static const uint32_t store_code[] = {
    0x3D400000, // addis r10, r0, 0
    0x614A5000, // ori   r10, r10, 0x5000
    0x3C80AABB, // addis r4, r0, 0xAABB
    0x6084CCDD, // ori   r4, r4, 0xCCDD
    0x38601234, // li    r3, 0x1234
    0x908A0000, // stw   r4, 0(r10)
    0x986A0004, // stb   r3, 4(r10)
    0xB06A0006, // sth   r3, 6(r10)
    0x908A0009, // stw   r4, 9(r10)    unaligned
    0xB08A000F, // sth   r4, 15(r10)   unaligned
    0x948A0020, // stwu  r4, 32(r10)
    0xB46A0004, // sthu  r3, 4(r10)
    0x9C6A0001, // stbu  r3, 1(r10)
    0x914A0008, // stw   r10, 8(r10)
    0x38600000, // li    r3, 0           may not erase state stores observed
    0x00000000, // illegal on purpose, the block stops short of it
};

constexpr uint32_t STORE_BASE = 0x6000;
constexpr uint32_t STORE_END  = STORE_BASE + uint32_t(sizeof(store_code) - 4);
constexpr uint32_t STORE_DATA = 0x5000;
constexpr int      STORE_WORDS = 16;

typedef struct StoreResult {
    AluResult regs;
    uint32_t  mem[STORE_WORDS];
} StoreResult;

static void load_store_code() {
    for (size_t i = 0; i < sizeof(store_code) / sizeof(store_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, STORE_BASE + uint32_t(i) * 4, store_code[i]);
    }
}

static StoreResult run_store_code() {
    for (int i = 0; i < STORE_WORDS; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, STORE_DATA + uint32_t(i) * 4, 0xE5E5E5E5u);
    }
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0x3C3C0000u | uint32_t(i);
    }
    ppc_state.pc  = STORE_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(STORE_END);

    StoreResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    for (int i = 0; i < STORE_WORDS; i++) {
        r.mem[i] = mmu_read_vmem<uint32_t>(NO_OPCODE, STORE_DATA + uint32_t(i) * 4);
    }
    return r;
}

static bool same_store(const StoreResult& a, const StoreResult& b, int* where) {
    int diff = -1;
    if (!same_alu(a.regs, b.regs, &diff)) { *where = diff; return false; }
    for (int i = 0; i < STORE_WORDS; i++) {
        if (a.mem[i] != b.mem[i]) { *where = 100 + i; return false; }
    }
    *where = -1;
    return true;
}

/** Comparing only the registers would not do for a store: its effect lives
    in memory, so the whole data area joins the comparison */
static void test_store_subset() {
    load_store_code();

    ppc_jit_disable();
    StoreResult interp = run_store_code();
    jit_check(interp.regs.pc == STORE_END, "the store program stopped where it should");

    ppc_jit_enable(JitBackend::threaded);
    StoreResult threaded = run_store_code();
    int where = -1;
    jit_check(same_store(interp, threaded, &where),
              "the threaded backend agrees with the interpreter on stores");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    StoreResult miss = run_store_code();
    StoreResult hit  = run_store_code();

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping the native comparison)" << endl;
        return;
    }

    where = -1;
    jit_check(same_store(interp, miss, &where), "emitted stores agree on the first pass");
    if (where >= 100) {
        cout << "    memory +0x" << hex << (where - 100) * 4 << ": interpreter 0x"
             << interp.mem[where - 100] << ", emitted 0x" << miss.mem[where - 100]
             << dec << endl;
    } else if (where >= 0) {
        cout << "    r" << where << ": interpreter 0x" << hex << interp.regs.gpr[where]
             << ", emitted 0x" << miss.regs.gpr[where] << dec << endl;
    }

    where = -1;
    jit_check(same_store(interp, hit, &where), "and agree with the TLB already warm");
}

/*  One CR field per instruction, covering the four comparators, andi. and
    andis., and the Rc forms of the opcodes already emitted. Runs twice, with
    XER[SO] clear and set, because that bit is copied into the field and a
    bug in it only shows while it is on.  */
static const uint32_t cr_code[] = {
    0x38600005, // li     r3, 5
    0x3880FFF9, // li     r4, -7
    0x2C030005, // cmpi   cr0, r3, 5
    0x2C830009, // cmpi   cr1, r3, 9
    0x2F840000, // cmpi   cr7, r4, 0
    0x29040001, // cmpli  cr2, r4, 1     sem sinal, -7 e enorme
    0x2E848000, // cmpi   cr5, r4, -32768  imediato fora de int8
    0x2B04FFFF, // cmpli  cr6, r4, 65535   imediato fora de int8
    0x7D832000, // cmp    cr3, r3, r4
    0x7E032040, // cmpl   cr4, r3, r4
    0x70650004, // andi.  r5, r3, 0x4
    0x7486FFFF, // andis. r6, r4, 0xFFFF
    0x7CE32215, // add.   r7, r3, r4
    0x7C682379, // or.    r8, r3, r4
    0x7C692039, // and.   r9, r3, r4
    0x7C8A0775, // extsb. r10, r4
    0x548B2037, // rlwinm. r11, r4, 4, 0, 27
    0x50852707, // rlwimi. r5, r4, 4, 28, 3   wrapping mask
    0x00000000, // illegal on purpose, the block stops short of it
};

constexpr uint32_t CR_BASE = 0x7000;
constexpr uint32_t CR_END  = CR_BASE + uint32_t(sizeof(cr_code) - 4);

typedef struct CrResult {
    AluResult regs;
    uint32_t  cr;
} CrResult;

static void load_cr_code() {
    for (size_t i = 0; i < sizeof(cr_code) / sizeof(cr_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, CR_BASE + uint32_t(i) * 4, cr_code[i]);
    }
}

static CrResult run_cr_code(bool set_summary_overflow) {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0x7B7B0000u | uint32_t(i);
    }
    ppc_state.cr = 0x0F0F0F0F;
    ppc_state.spr[SPR::XER] = set_summary_overflow ? XER::SO : 0;
    ppc_state.pc  = CR_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(CR_END);

    CrResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.cr           = ppc_state.cr;
    return r;
}

static bool same_cr(const CrResult& a, const CrResult& b, int* where) {
    if (a.cr != b.cr) { *where = 200; return false; }
    return same_alu(a.regs, b.regs, where);
}

static void test_cr_subset() {
    load_cr_code();

    for (int pass = 0; pass < 2; pass++) {
        const bool so = pass != 0;

        ppc_jit_disable();
        CrResult interp = run_cr_code(so);
        jit_check(interp.regs.pc == CR_END,
                  so ? "the CR program stopped where it should, with XER[SO]"
                     : "the CR program stopped where it should");

        ppc_jit_enable(JitBackend::threaded);
        int where = -1;
        CrResult threaded = run_cr_code(so);
        jit_check(same_cr(interp, threaded, &where),
                  so ? "the threaded backend agrees on CR, with XER[SO]"
                     : "the threaded backend agrees on CR");

        ppc_jit_disable();
        ppc_jit_enable(JitBackend::automatic);
        CrResult native = run_cr_code(so);

        if (ppc_jit_native_compiles() == 0) {
            cout << "  (no emitter on this host, skipping the native comparison)" << endl;
            return;
        }

        where = -1;
        jit_check(same_cr(interp, native, &where),
                  so ? "emitted code agrees on CR, with XER[SO]"
                     : "emitted code agrees on CR");
        if (where == 200) {
            cout << "    cr: interpreter 0x" << hex << interp.cr
                 << ", emitted 0x" << native.cr << dec << endl;
        } else if (where >= 0) {
            cout << "    r" << where << ": interpreter 0x" << hex
                 << interp.regs.gpr[where] << ", emitted 0x"
                 << native.regs.gpr[where] << dec << endl;
        }
    }
}

/*  Function call and return: bl stacks into LR, blr goes back through it,
    bcctr jumps through CTR, with mflr, mtctr, mtlr and mfctr on the way, a
    conditional bclr not taken, and an eieio that decodes into nothing and
    only counts as retired.  */
static const uint32_t branch_code[] = {
    0x38600000, // +0x00 li     r3, 0
    0x4800001D, // +0x04 bl     +0x1C          -> +0x20
    0x38630001, // +0x08 addi   r3, r3, 1      <- blr comes back here
    0x3CC00000, // +0x0C addis  r6, r0, 0
    0x60C68038, // +0x10 ori    r6, r6, 0x8038
    0x7CC903A6, // +0x14 mtctr  r6
    0x4E800420, // +0x18 bcctr  20, 0          -> 0x8038
    0x00000000, // +0x1C illegal, never runs
    0x38800005, // +0x20 li     r4, 5
    0x2C040063, // +0x24 cmpi   cr0, r4, 99    EQ clear
    0x7CA802A6, // +0x28 mflr   r5             r5 = 0x8008
    0x4D820020, // +0x2C beqlr                 not taken
    0x4E800020, // +0x30 blr                   -> 0x8008
    0x00000000, // +0x34 illegal, never runs
    0x7CC803A6, // +0x38 mtlr   r6
    0x7D0902A6, // +0x3C mfctr  r8             r8 = 0x8038
    0x7C0006AC, // +0x40 eieio
    0x38E00007, // +0x44 li     r7, 7
    0x00000000, // +0x48 illegal, the program stops here
};

constexpr uint32_t BR_BASE = 0x8000;
constexpr uint32_t BR_END  = BR_BASE + 0x48;
constexpr uint64_t BR_RETIRED = 16;

typedef struct SprResult {
    AluResult regs;
    uint32_t  lr, ctr, cr;
} SprResult;

static void load_branch_code() {
    for (size_t i = 0; i < sizeof(branch_code) / sizeof(branch_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, BR_BASE + uint32_t(i) * 4, branch_code[i]);
    }
}

static SprResult run_branch_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xC3C30000u | uint32_t(i);
    }
    ppc_state.spr[SPR::LR]  = 0;
    ppc_state.spr[SPR::CTR] = 0;
    ppc_state.cr  = 0;
    ppc_state.pc  = BR_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(BR_END);

    SprResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.lr  = ppc_state.spr[SPR::LR];
    r.ctr = ppc_state.spr[SPR::CTR];
    r.cr  = ppc_state.cr;
    return r;
}

static bool same_spr(const SprResult& a, const SprResult& b, int* where) {
    if (a.lr  != b.lr)  { *where = 300; return false; }
    if (a.ctr != b.ctr) { *where = 301; return false; }
    if (a.cr  != b.cr)  { *where = 302; return false; }
    return same_alu(a.regs, b.regs, where);
}

static void report_spr(const SprResult& interp, const SprResult& got, int where,
                       const char* label) {
    if (where == 300) {
        cout << "    lr: interpreter 0x" << hex << interp.lr
             << ", " << label << " 0x" << got.lr << dec << endl;
    } else if (where == 301) {
        cout << "    ctr: interpreter 0x" << hex << interp.ctr
             << ", " << label << " 0x" << got.ctr << dec << endl;
    } else if (where == 302) {
        cout << "    cr: interpreter 0x" << hex << interp.cr
             << ", " << label << " 0x" << got.cr << dec << endl;
    } else if (where >= 0) {
        cout << "    r" << where << ": interpreter 0x" << hex
             << interp.regs.gpr[where] << ", " << label << " 0x"
             << got.regs.gpr[where] << dec << endl;
    }
}

/** Emitted bclr and bcctr have to land exactly where the interpreter
    lands, with the same LR, the same CTR and the same retired count */
static void test_branch_subset() {
    load_branch_code();

    ppc_jit_disable();
    SprResult interp = run_branch_code();
    jit_check(interp.regs.pc == BR_END, "the branch program stopped where it should");
    jit_check(interp.regs.retired == BR_RETIRED,
              "the interpreter retired the expected count in the branch program");

    ppc_jit_enable(JitBackend::threaded);
    SprResult threaded = run_branch_code();
    int where = -1;
    jit_check(same_spr(interp, threaded, &where),
              "the threaded backend agrees on the branch program");
    report_spr(interp, threaded, where, "threaded");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    SprResult native = run_branch_code();

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping the native comparison)" << endl;
        return;
    }

    where = -1;
    jit_check(same_spr(interp, native, &where),
              "emitted code agrees on the branch program");
    report_spr(interp, native, where, "emitted");
    jit_check(ppc_jit_threaded_compiles() == 0,
              "the emitter took every branch block instead of declining");

    SprResult again = run_branch_code();
    where = -1;
    jit_check(same_spr(interp, again, &where),
              "a second pass over the cached branch blocks agrees");
}

/*  Superblock formation: a taken forward beq leaving through a side exit, a
    not taken one falling through it, a bcl 20,31,$+4 dissolving into an LR
    store, and unconditional skips walked through with gaps behind them. The
    retired count is the sharp edge here: every walked through branch still
    retires, and a side exit taken must retire exactly up to itself.  */
static const uint32_t superblock_code[] = {
    0x38600000, // +0x00 li     r3, 0
    0x2C030000, // +0x04 cmpwi  cr0, r3, 0     EQ set
    0x41820010, // +0x08 beq    +0x10          -> +0x18, taken side exit
    0x3860000B, // +0x0C li     r3, 11         skipped; must not erase first r3
    0x38A00016, // +0x10 li     r5, 22         skipped
    0x4800000C, // +0x14 b      +0x0C          -> +0x20, never runs
    0x38C00021, // +0x18 li     r6, 33
    0x2C060063, // +0x1C cmpwi  cr0, r6, 99    EQ clear
    0x41820008, // +0x20 beq    +0x08          -> +0x28, not taken
    0x38E0002C, // +0x24 li     r7, 44
    0x429F0005, // +0x28 bcl    20, 31, +4     LR = +0x2C, walked through
    0x7D0802A6, // +0x2C mflr   r8             r8 = +0x2C
    0x48000008, // +0x30 b      +0x08          -> +0x38, walked through
    0x39200037, // +0x34 li     r9, 55         skipped, lives in a gap
    0x39400042, // +0x38 li     r10, 66
    0x00000000, // +0x3C illegal, the program stops here
};

constexpr uint32_t SB_BASE = 0x8800;
constexpr uint32_t SB_END  = SB_BASE + 0x3C;
constexpr uint64_t SB_RETIRED = 11;

static void load_superblock_code() {
    for (size_t i = 0; i < sizeof(superblock_code) / sizeof(superblock_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, SB_BASE + uint32_t(i) * 4,
                                 superblock_code[i]);
    }
}

static SprResult run_superblock_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xC3C30000u | uint32_t(i);
    }
    ppc_state.spr[SPR::LR]  = 0;
    ppc_state.spr[SPR::CTR] = 0;
    ppc_state.cr  = 0;
    ppc_state.pc  = SB_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(SB_END);

    SprResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.lr  = ppc_state.spr[SPR::LR];
    r.ctr = ppc_state.spr[SPR::CTR];
    r.cr  = ppc_state.cr;
    return r;
}

static void test_superblock_subset() {
    load_superblock_code();

    ppc_jit_disable();
    SprResult interp = run_superblock_code();
    jit_check(interp.regs.pc == SB_END,
              "the superblock program stopped where it should");
    jit_check(interp.regs.retired == SB_RETIRED,
              "the interpreter retired the expected count in the superblock program");
    jit_check(interp.regs.gpr[3] == 0,
              "the taken side exit preserved the GPR value stored before it");

    ppc_jit_enable(JitBackend::threaded);
    SprResult threaded = run_superblock_code();
    int where = -1;
    jit_check(same_spr(interp, threaded, &where),
              "the threaded backend agrees on the superblock program");
    report_spr(interp, threaded, where, "threaded");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    SprResult native = run_superblock_code();

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping the native comparison)" << endl;
        return;
    }

    where = -1;
    jit_check(same_spr(interp, native, &where),
              "emitted code agrees on the superblock program");
    report_spr(interp, native, where, "emitted");

    SprResult again = run_superblock_code();
    where = -1;
    jit_check(same_spr(interp, again, &where),
              "a second pass over the cached superblocks agrees");
}

/*  A walked through branch landing exactly on the block budget edge. With
    the budget at two, the b occupies the last slot; a translator that walks
    through it anyway ends the block with nothing decoded at the target, and
    the fall off the end exit resumes at the branch's fall through, running
    the very code the branch jumped over. The r9 poison in the jumped over
    region is the detector.  */
static const uint32_t budget_edge_code[] = {
    0x38600001, // +0x00 li r3, 1
    0x48000008, // +0x04 b  +0x08       -> +0x0C
    0x39200063, // +0x08 li r9, 99      jumped over, must never run
    0x38800002, // +0x0C li r4, 2
    0x00000000, // +0x10 illegal, the program stops here
};

constexpr uint32_t BE_BASE = 0x8900;
constexpr uint32_t BE_END  = BE_BASE + 0x10;

static void test_superblock_budget_edge() {
    for (size_t i = 0; i < sizeof(budget_edge_code) / sizeof(budget_edge_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, BE_BASE + uint32_t(i) * 4,
                                 budget_edge_code[i]);
    }

    const uint32_t saved_budget = dppc_jit::jit_max_block_insns;
    dppc_jit::jit_max_block_insns = 2;

    ppc_jit_disable();
    ppc_state.pc = BE_BASE;
    for (int i = 0; i < 32; i++) ppc_state.gpr[i] = 0;
    g_icycles = 0; g_icycles_max = 0; exec_flags = 0; power_on = true;
    ppc_exec_until(BE_END);
    const uint32_t interp_r9 = ppc_state.gpr[9];
    const uint32_t interp_pc = ppc_state.pc;

    ppc_jit_enable(JitBackend::threaded);
    ppc_state.pc = BE_BASE;
    for (int i = 0; i < 32; i++) ppc_state.gpr[i] = 0;
    g_icycles = 0; g_icycles_max = 0; exec_flags = 0; power_on = true;
    ppc_exec_until(BE_END);
    jit_check(ppc_state.pc == interp_pc && ppc_state.gpr[9] == interp_r9,
              "a walk through on the budget edge stays off the jumped over code (threaded)");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    ppc_state.pc = BE_BASE;
    for (int i = 0; i < 32; i++) ppc_state.gpr[i] = 0;
    g_icycles = 0; g_icycles_max = 0; exec_flags = 0; power_on = true;
    ppc_exec_until(BE_END);
    jit_check(ppc_state.pc == interp_pc && ppc_state.gpr[9] == interp_r9,
              "a walk through on the budget edge stays off the jumped over code (emitted)");

    dppc_jit::jit_max_block_insns = saved_budget;
    ppc_jit_disable();
}

/*  A call into another page, walked through into one block guarded by an
    ItransGuard at the seam. The walked block must agree with the interpreter
    and expose the caller's GPR state past the guard. A store into the CALLEE
    page must kill it through its second invalidation range, and a flushed
    ITLB entry must make the seam guard fail into an honest fetch instead of
    running stale.  */
static const uint32_t cross_caller_code[] = {
    0x38600005, // 0xA000 li   r3, 5
    0x48000FFD, // 0xA004 bl   -> 0xB000
    0x38C00009, // 0xA008 li   r6, 9          <- blr comes back here
    0x00000000, // 0xA00C illegal, the program stops here
};
static const uint32_t cross_callee_code[] = {
    0x7C641B78, // 0xB000 mr   r4, r3           guard must expose caller's r3
    0x38600007, // 0xB004 li   r3, 7            overwrite only after the guard
    0x7CA802A6, // 0xB008 mflr r5               r5 = 0xA008
    0x4E800020, // 0xB00C blr
};

constexpr uint32_t CC_BASE   = 0xA000;
constexpr uint32_t CC_CALLEE = 0xB000;
constexpr uint32_t CC_END    = CC_BASE + 0x0C;
constexpr uint64_t CC_RETIRED = 7;

static void load_cross_code() {
    for (size_t i = 0; i < sizeof(cross_caller_code) / 4; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, CC_BASE + uint32_t(i) * 4,
                                 cross_caller_code[i]);
    }
    for (size_t i = 0; i < sizeof(cross_callee_code) / 4; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, CC_CALLEE + uint32_t(i) * 4,
                                 cross_callee_code[i]);
    }
}

static SprResult run_cross_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xC3C30000u | uint32_t(i);
    }
    ppc_state.spr[SPR::LR]  = 0;
    ppc_state.spr[SPR::CTR] = 0;
    ppc_state.cr  = 0;
    ppc_state.pc  = CC_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(CC_END);

    SprResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.lr  = ppc_state.spr[SPR::LR];
    r.ctr = ppc_state.spr[SPR::CTR];
    r.cr  = ppc_state.cr;
    return r;
}

static void test_cross_page_call() {
    load_cross_code();

    ppc_jit_disable();
    SprResult interp = run_cross_code();
    jit_check(interp.regs.pc == CC_END && interp.regs.retired == CC_RETIRED &&
              interp.regs.gpr[4] == 5 && interp.regs.gpr[3] == 7,
              "the cross page call program behaves on the interpreter");

    ppc_jit_enable(JitBackend::threaded);
    SprResult threaded = run_cross_code();
    int where = -1;
    jit_check(same_spr(interp, threaded, &where),
              "the threaded backend agrees on the cross page call");
    report_spr(interp, threaded, where, "threaded");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    SprResult native = run_cross_code();
    where = -1;
    jit_check(same_spr(interp, native, &where),
              "emitted code agrees on the cross page call");
    report_spr(interp, native, where, "emitted");

    if (ppc_jit_native_compiles() == 0) {
        return; // no emitter on this host, the rest probes emitted paths
    }

    // a store into the CALLEE page has to reach the walked through block
    // through its second invalidation range: rerunning must see the new
    // value, and a stale 5 here means the range dangled
    mmu_write_vmem<uint32_t>(NO_OPCODE, CC_CALLEE, 0x38800008); // li r4, 8
    SprResult patched = run_cross_code();
    jit_check(patched.regs.gpr[4] == 8 && patched.regs.retired == CC_RETIRED,
              "a store into the callee page invalidates the walked through block");

    // with the callee's ITLB entry flushed, the seam guard must fail into
    // an honest fetch and still come out with the interpreter's answer
    mmu_write_vmem<uint32_t>(NO_OPCODE, CC_CALLEE, 0x7C641B78); // mr r4, r3
    tlb_flush_entry(CC_CALLEE);
    SprResult reguarded = run_cross_code();
    where = -1;
    jit_check(same_spr(interp, reguarded, &where),
              "a failed seam guard falls back to an honest fetch");
    report_spr(interp, reguarded, where, "reguarded");

    // the blr of a walked leaf dissolves only while LR provably holds the
    // call's return address; squeezing the budget so the blr lands outside
    // it and then exactly on its last slot must refuse the dissolve and
    // stay correct through the ordinary endings
    const uint32_t saved_budget = dppc_jit::jit_max_block_insns;
    for (uint32_t budget = 4; budget <= 5; budget++) {
        dppc_jit::jit_max_block_insns = budget;
        ppc_jit_disable();
        ppc_jit_enable(JitBackend::automatic);
        SprResult squeezed = run_cross_code();
        where = -1;
        jit_check(same_spr(interp, squeezed, &where),
                  "a budget squeezed leaf return stays exact");
        report_spr(interp, squeezed, where, "squeezed");
    }
    dppc_jit::jit_max_block_insns = saved_budget;
    ppc_jit_disable();
}

/*  A leaf that redirects its own return with mtlr: the dissolve must see
    the LR write and let the blr end the block, or execution would fall
    into the very instruction the leaf jumped over.  */
static const uint32_t lrw_caller_code[] = {
    0x38600005, // 0xC000 li   r3, 5
    0x48000FFD, // 0xC004 bl   -> 0xD000
    0x38C00009, // 0xC008 li   r6, 9          the redirected return skips this
    0x38E00003, // 0xC00C li   r7, 3          and lands here
    0x00000000, // 0xC010 illegal, the program stops here
};
static const uint32_t lrw_callee_code[] = {
    0x7CA802A6, // 0xD000 mflr r5             r5 = 0xC008
    0x38850004, // 0xD004 addi r4, r5, 4      r4 = 0xC00C
    0x7C8803A6, // 0xD008 mtlr r4
    0x4E800020, // 0xD00C blr                 -> 0xC00C
};

constexpr uint32_t LRW_BASE   = 0xC000;
constexpr uint32_t LRW_CALLEE = 0xD000;
constexpr uint32_t LRW_END    = LRW_BASE + 0x10;

static SprResult run_lr_write_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xC3C30000u | uint32_t(i);
    }
    ppc_state.spr[SPR::LR]  = 0;
    ppc_state.spr[SPR::CTR] = 0;
    ppc_state.cr  = 0;
    ppc_state.pc  = LRW_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(LRW_END);

    SprResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.lr  = ppc_state.spr[SPR::LR];
    r.ctr = ppc_state.spr[SPR::CTR];
    r.cr  = ppc_state.cr;
    return r;
}

static void test_leaf_lr_write() {
    for (size_t i = 0; i < sizeof(lrw_caller_code) / 4; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, LRW_BASE + uint32_t(i) * 4,
                                 lrw_caller_code[i]);
    }
    for (size_t i = 0; i < sizeof(lrw_callee_code) / 4; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, LRW_CALLEE + uint32_t(i) * 4,
                                 lrw_callee_code[i]);
    }

    ppc_jit_disable();
    SprResult interp = run_lr_write_code();
    jit_check(interp.regs.pc == LRW_END && interp.regs.retired == 7 &&
              interp.regs.gpr[6] == (0xC3C30000u | 6) && interp.regs.gpr[7] == 3,
              "the mtlr leaf program behaves on the interpreter");

    ppc_jit_enable(JitBackend::threaded);
    SprResult threaded = run_lr_write_code();
    int where = -1;
    jit_check(same_spr(interp, threaded, &where),
              "a leaf that rewrites LR keeps its blr honest (threaded)");
    report_spr(interp, threaded, where, "threaded");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    SprResult native = run_lr_write_code();
    where = -1;
    jit_check(same_spr(interp, native, &where),
              "a leaf that rewrites LR keeps its blr honest (emitted)");
    report_spr(interp, native, where, "emitted");
    ppc_jit_disable();
}

/*  Two calls to the same leaf in a row: the first pair dissolves, the walk
    machinery is spent for the block, and the second call has to end it and
    chain through the ordinary path with exact retirement.  */
static const uint32_t twice_caller_code[] = {
    0x38600005, // 0xE000 li   r3, 5
    0x48000FFD, // 0xE004 bl   -> 0xF000
    0x48000FF9, // 0xE008 bl   -> 0xF000
    0x38C00009, // 0xE00C li   r6, 9
    0x00000000, // 0xE010 illegal, the program stops here
};
static const uint32_t twice_callee_code[] = {
    0x38800007, // 0xF000 li   r4, 7
    0x7CA802A6, // 0xF004 mflr r5
    0x4E800020, // 0xF008 blr
};

constexpr uint32_t TW_BASE   = 0xE000;
constexpr uint32_t TW_CALLEE = 0xF000;
constexpr uint32_t TW_END    = TW_BASE + 0x10;

static SprResult run_twice_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xC3C30000u | uint32_t(i);
    }
    ppc_state.spr[SPR::LR]  = 0;
    ppc_state.spr[SPR::CTR] = 0;
    ppc_state.cr  = 0;
    ppc_state.pc  = TW_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(TW_END);

    SprResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.lr  = ppc_state.spr[SPR::LR];
    r.ctr = ppc_state.spr[SPR::CTR];
    r.cr  = ppc_state.cr;
    return r;
}

static void test_leaf_called_twice() {
    for (size_t i = 0; i < sizeof(twice_caller_code) / 4; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, TW_BASE + uint32_t(i) * 4,
                                 twice_caller_code[i]);
    }
    for (size_t i = 0; i < sizeof(twice_callee_code) / 4; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, TW_CALLEE + uint32_t(i) * 4,
                                 twice_callee_code[i]);
    }

    ppc_jit_disable();
    SprResult interp = run_twice_code();
    jit_check(interp.regs.pc == TW_END && interp.regs.retired == 10 &&
              interp.lr == TW_BASE + 0x0C,
              "the double call program behaves on the interpreter");

    ppc_jit_enable(JitBackend::threaded);
    SprResult threaded = run_twice_code();
    int where = -1;
    jit_check(same_spr(interp, threaded, &where),
              "two calls into the same leaf stay exact (threaded)");
    report_spr(interp, threaded, where, "threaded");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    SprResult native = run_twice_code();
    where = -1;
    jit_check(same_spr(interp, native, &where),
              "two calls into the same leaf stay exact (emitted)");
    report_spr(interp, native, where, "emitted");
    ppc_jit_disable();
}

/*  The compare and branch fusion: a branch may ride the FLAGS of the
    compare that wrote its CR field, but only while nothing in between
    scratched them. The add after the first compare leaves FLAGS that say
    the opposite of the CR, which is the trap a missed invalidation falls
    into, and the second compare feeds two branches in a row, which is the
    fall through side keeping the FLAGS alive.  */
static const uint32_t fuse_code[] = {
    0x38600005, // 0x8300 li    r3, 5
    0x38800009, // 0x8304 li    r4, 9
    0x7C032000, // 0x8308 cmpw  cr0, r3, r4     LT
    0x7CA42214, // 0x830C add   r5, r4, r4      FLAGS now say the opposite
    0x41800008, // 0x8310 blt   -> 0x8318       taken off the CR, not the add
    0x38C00001, // 0x8314 li    r6, 1           jumped over
    0x38E00002, // 0x8318 li    r7, 2
    0x7C041800, // 0x831C cmpw  cr0, r4, r3     GT
    0x4180000C, // 0x8320 blt   -> 0x832C       not taken, FLAGS stay live
    0x41810008, // 0x8324 bgt   -> 0x832C       taken off the same compare
    0x39000003, // 0x8328 li    r8, 3           jumped over
    // the shape the ROM nap gate broke on: a compare into cr7, a dot form
    // rewriting cr0, a branch on cr7 whose reload test scratches the FLAGS,
    // then a branch on cr0 that must not ride them
    0x39200004, // 0x832C li    r9, 4
    0x2F890007, // 0x8330 cmpwi cr7, r9, 7      LT, EQ clear
    0x7C6A0379, // 0x8334 or.   r10, r3, r0     r10 negative, cr0 = LT
    0x419E000C, // 0x8338 beq   cr7 -> 0x8344   not taken, reload test runs
    0x41800010, // 0x833C blt   cr0 -> 0x834C   taken off cr0, not the test
    0x39600006, // 0x8340 li    r11, 6          jumped over
    0x39800007, // 0x8344 li    r12, 7          jumped over
    0x39A00008, // 0x8348 li    r13, 8          jumped over
    0x00000000, // 0x834C illegal, the program stops here
};

constexpr uint32_t FUSE_BASE = 0x8300;
constexpr uint32_t FUSE_END  = FUSE_BASE + 0x4C;

static SprResult run_fuse_code() {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xC3C30000u | uint32_t(i);
    }
    ppc_state.spr[SPR::LR]  = 0;
    ppc_state.spr[SPR::CTR] = 0;
    ppc_state.cr  = 0;
    ppc_state.pc  = FUSE_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(FUSE_END);

    SprResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.lr  = ppc_state.spr[SPR::LR];
    r.ctr = ppc_state.spr[SPR::CTR];
    r.cr  = ppc_state.cr;
    return r;
}

/*  Every fusible shape, exhaustively: branch on bit set and clear, all
    three testable bits, both compare signednesses and every outcome of
    the compare, as a forward side exit off a fused compare.  */
static void test_cr_fusion_matrix() {
    static const uint32_t operands[3][2] = {
        {5, 9},          // lt
        {7, 7},          // eq
        {0x80000000, 2}, // gt signed, lt unsigned
    };

    for (int form = 0; form < 4; form++) {  // rr, rr cr7, imm, imm lk
        for (int uns = 0; uns <= 1; uns++) {
        for (int bo = 4; bo <= 12; bo += 8) {
            for (int bi = 0; bi <= 2; bi++) {
                for (int val = 0; val < 3; val++) {
                    uint32_t cmp_op, bc_bi = uint32_t(bi);
                    switch (form) {
                    case 0: // register form on cr0
                        cmp_op = uns ? 0x7C032040u : 0x7C032000u;
                        break;
                    case 1: // register form on cr7
                        cmp_op = (uns ? 0x7C032040u : 0x7C032000u) | (7u << 23);
                        bc_bi += 28;
                        break;
                    default: // immediate form, r3 against 7
                        cmp_op = (uns ? 0x28030000u : 0x2C030000u) | 7u;
                        break;
                    }
                    const uint32_t bc_op = 0x40000008u |
                        (uint32_t(bo) << 21) | (bc_bi << 16) |
                        (form == 3 ? 1u : 0u);
                    const uint32_t prog[] = {
                        cmp_op,
                        bc_op,          // forward side exit over one word
                        0x38C00001,     // li r6, 1, the jumped over word
                        0x38E00002,     // li r7, 2
                        0x00000000,     // illegal, the program stops here
                    };
                    for (size_t i = 0; i < sizeof(prog) / 4; i++) {
                        mmu_write_vmem<uint32_t>(NO_OPCODE,
                            FUSE_BASE + 0x40 + uint32_t(i) * 4, prog[i]);
                    }

                    SprResult res[3];
                    for (int run = 0; run < 3; run++) {
                        ppc_jit_disable();
                        if (run == 1) ppc_jit_enable(JitBackend::threaded);
                        if (run == 2) ppc_jit_enable(JitBackend::automatic);
                        for (int i = 0; i < 32; i++)
                            ppc_state.gpr[i] = 0xC3C30000u | uint32_t(i);
                        ppc_state.gpr[3] = operands[val][0];
                        ppc_state.gpr[4] = operands[val][1];
                        ppc_state.spr[SPR::LR]  = 0;
                        ppc_state.spr[SPR::CTR] = 0;
                        ppc_state.cr  = 0;
                        ppc_state.pc  = FUSE_BASE + 0x40;
                        g_icycles     = 0;
                        g_icycles_max = 0;
                        exec_flags    = 0;
                        power_on      = true;
                        ppc_exec_until(FUSE_BASE + 0x50);
                        for (int i = 0; i < 32; i++)
                            res[run].regs.gpr[i] = ppc_state.gpr[i];
                        res[run].regs.pc      = ppc_state.pc;
                        res[run].regs.retired = g_icycles;
                        res[run].lr  = ppc_state.spr[SPR::LR];
                        res[run].ctr = ppc_state.spr[SPR::CTR];
                        res[run].cr  = ppc_state.cr;
                    }
                    ppc_jit_disable();

                    int where = -1;
                    if (!same_spr(res[0], res[1], &where) ||
                        !same_spr(res[0], res[2], &where)) {
                        std::printf("  fusion matrix: form=%d uns=%d bo=%d bi=%d val=%d "
                                    "r6 int=%08X thr=%08X emit=%08X pc %08X/%08X/%08X\n",
                                    form, uns, bo, bi, val,
                                    res[0].regs.gpr[6], res[1].regs.gpr[6],
                                    res[2].regs.gpr[6], res[0].regs.pc,
                                    res[1].regs.pc, res[2].regs.pc);
                        jit_check(false, "a fusible shape diverged");
                    }
                }
            }
        }
        }
    }
    jit_check(true, "every fusible compare and branch shape agrees");
}

static void test_cr_fusion() {
    for (size_t i = 0; i < sizeof(fuse_code) / 4; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, FUSE_BASE + uint32_t(i) * 4,
                                 fuse_code[i]);
    }

    ppc_jit_disable();
    SprResult interp = run_fuse_code();
    jit_check(interp.regs.pc == FUSE_END && interp.regs.retired == 14 &&
              interp.regs.gpr[6] == (0xC3C30000u | 6) &&
              interp.regs.gpr[8] == (0xC3C30000u | 8) &&
              interp.regs.gpr[11] == (0xC3C30000u | 11) &&
              interp.regs.gpr[7] == 2 && interp.regs.gpr[9] == 4 &&
              interp.regs.gpr[10] == 0xC3C30005u,
              "the fusion program behaves on the interpreter");

    ppc_jit_enable(JitBackend::threaded);
    SprResult threaded = run_fuse_code();
    int where = -1;
    jit_check(same_spr(interp, threaded, &where),
              "compare and branch agree with the interpreter (threaded)");
    report_spr(interp, threaded, where, "threaded");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    SprResult native = run_fuse_code();
    where = -1;
    jit_check(same_spr(interp, native, &where),
              "a branch riding the compare's FLAGS stays exact (emitted)");
    report_spr(interp, native, where, "emitted");
    ppc_jit_disable();
}

/*  The XER[CA] family: addc, adde, addic, addic., subfc, subfe, subfic,
    addze and subf, with the carry chaining from one into the next and the
    forms with both operands in the same register, which are the ones that
    break a careless emitter.  */
static const uint32_t carry_code[] = {
    0x3C60FFFF, // addis  r3, r0, 0xFFFF
    0x6063FFFF, // ori    r3, r3, 0xFFFF     r3 = 0xFFFFFFFF
    0x38800001, // li     r4, 1
    0x7CA32014, // addc   r5, r3, r4         0, CA=1
    0x7CC42114, // adde   r6, r4, r4         3, CA=0
    0x30E30002, // addic  r7, r3, 2          1, CA=1
    0x3504FFFF, // addic. r8, r4, -1         0, CA=1, CR0=EQ
    0x7D241810, // subfc  r9, r4, r3         0xFFFFFFFE, CA=1
    0x7D432110, // subfe  r10, r3, r4        2, CA=0
    0x2163FFFF, // subfic r11, r3, -1        0, CA=1
    0x7D830194, // addze  r12, r3            0, CA=1
    0x7DA31914, // adde   r13, r3, r3        0xFFFFFFFF, CA=1
    0x7DC41850, // subf   r14, r4, r3        0xFFFFFFFE, CA stays
    0x7DEF7910, // subfe  r15, r15, r15      both operands in one register
    0x7E108014, // addc   r16, r16, r16      likewise
    0x00000000, // illegal on purpose, the program stops here
};

constexpr uint32_t CARRY_BASE = 0x8100;
constexpr uint32_t CARRY_END  = CARRY_BASE + uint32_t(sizeof(carry_code) - 4);

typedef struct CarryResult {
    AluResult regs;
    uint32_t  xer, cr;
} CarryResult;

static void load_carry_code() {
    for (size_t i = 0; i < sizeof(carry_code) / sizeof(carry_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, CARRY_BASE + uint32_t(i) * 4, carry_code[i]);
    }
}

static CarryResult run_carry_code(bool carry_in) {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xA5A50000u | uint32_t(i);
    }
    ppc_state.cr = 0;
    ppc_state.spr[SPR::XER] = carry_in ? XER::CA : 0;
    ppc_state.pc  = CARRY_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(CARRY_END);

    CarryResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.xer          = ppc_state.spr[SPR::XER];
    r.cr           = ppc_state.cr;
    return r;
}

static bool same_carry(const CarryResult& a, const CarryResult& b, int* where) {
    if (a.xer != b.xer) { *where = 400; return false; }
    if (a.cr  != b.cr)  { *where = 401; return false; }
    return same_alu(a.regs, b.regs, where);
}

static void report_carry(const CarryResult& interp, const CarryResult& got,
                         int where, const char* label) {
    if (where == 400) {
        cout << "    xer: interpreter 0x" << hex << interp.xer
             << ", " << label << " 0x" << got.xer << dec << endl;
    } else if (where == 401) {
        cout << "    cr: interpreter 0x" << hex << interp.cr
             << ", " << label << " 0x" << got.cr << dec << endl;
    } else if (where >= 0) {
        cout << "    r" << where << ": interpreter 0x" << hex
             << interp.regs.gpr[where] << ", " << label << " 0x"
             << got.regs.gpr[where] << dec << endl;
    }
}

/** Emitted carry uses the host flag while the interpreter compares by
    hand, and the two have to agree in every register and in XER itself,
    with the carry coming in both clear and set */
static void test_carry_subset() {
    load_carry_code();

    for (int pass = 0; pass < 2; pass++) {
        const bool ca = pass != 0;

        ppc_jit_disable();
        CarryResult interp = run_carry_code(ca);
        jit_check(interp.regs.pc == CARRY_END,
                  ca ? "the carry program stopped where it should, with CA"
                     : "the carry program stopped where it should");

        ppc_jit_enable(JitBackend::threaded);
        CarryResult threaded = run_carry_code(ca);
        int where = -1;
        jit_check(same_carry(interp, threaded, &where),
                  ca ? "the threaded backend agrees on carry, with CA"
                     : "the threaded backend agrees on carry");
        report_carry(interp, threaded, where, "threaded");

        ppc_jit_disable();
        ppc_jit_enable(JitBackend::automatic);
        CarryResult native = run_carry_code(ca);

        if (ppc_jit_native_compiles() == 0) {
            cout << "  (no emitter on this host, skipping the native comparison)" << endl;
            return;
        }

        where = -1;
        jit_check(same_carry(interp, native, &where),
                  ca ? "emitted code agrees on carry, with CA"
                     : "emitted code agrees on carry");
        report_carry(interp, native, where, "emitted");
        jit_check(ppc_jit_threaded_compiles() == 0,
                  "the emitter took every carry block instead of declining");
    }
}

/*  The OE forms, the multiply family and mtcrf. Overflows and clean results
    alternate so that OV is seen both raised and cleared while SO sticks, the
    CA consumers run against a CA a previous instruction produced, and the
    forms with every operand in one register cover the aliasing rules of the
    emitter. The second pass starts with SO, OV and CA already set.  */
static const uint32_t ov_code[] = {
    0x3C607FFF, // addis  r3, r0, 0x7FFF
    0x6063FFFF, // ori    r3, r3, 0xFFFF     r3 = 0x7FFFFFFF
    0x38800001, // li     r4, 1
    0x39A0FFFE, // li     r13, -2
    0x7CA32614, // addo   r5, r3, r4         overflow into 0x80000000
    0x7CC32415, // addco. r6, r3, r4         carry and overflow together
    0x7CE42614, // addo   r7, r4, r4         clean, OV falls, SO stays
    0x7D0319D6, // mullwo r8, r3, r3         the product loses its top half
    0x7D2425D7, // mullwo. r9, r4, r4        clean, with CR0
    0x1D43FFFE, // mulli  r10, r3, -2
    0x7D631896, // mulhw  r11, r3, r3
    0x7D836816, // mulhwu r12, r3, r13       unsigned reading of -2
    0x7DC36896, // mulhw  r14, r3, r13       signed reading of the same
    0x7DE41C10, // subfco r15, r4, r3        CA set, no overflow
    0x7E431D14, // addeo  r18, r3, r3        consumes that CA, overflows
    0x7E631D10, // subfeo r19, r3, r3        clean, CA falls
    0x7E850594, // addzeo r20, r5
    0x7E2504D0, // nego   r17, r5            the one negation that overflows
    0x7E0304D0, // nego   r16, r3            and the ordinary one after it
    0x7EA32C50, // subfo  r21, r3, r5        negative minus positive
    0x7ED6B5D6, // mullwo r22, r22, r22      dst and both operands aliased
    0x7EF7BD14, // addeo  r23, r23, r23      likewise through the carry path
    0x7F18C510, // subfeo r24, r24, r24      the flipped operand may not alias
    0x7F39C896, // mulhw  r25, r25, r25
    0x7E481120, // mtcrf  0x81, r18          two fields merged
    0x7DEFF120, // mtcrf  0xFF, r15          the full mask single store
    0x7C808120, // mtcrf  0x08, r4           one field, from a value with none
    0x00000000, // illegal on purpose, the program stops here
};

constexpr uint32_t OV_BASE = 0x8400;
constexpr uint32_t OV_END  = OV_BASE + uint32_t(sizeof(ov_code) - 4);

static void load_ov_code() {
    for (size_t i = 0; i < sizeof(ov_code) / sizeof(ov_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, OV_BASE + uint32_t(i) * 4, ov_code[i]);
    }
}

static CarryResult run_ov_code(uint32_t xer_in) {
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0xA5A50000u | uint32_t(i);
    }
    ppc_state.cr = 0x0F0F0F0F;
    ppc_state.spr[SPR::XER] = xer_in;
    ppc_state.pc  = OV_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(OV_END);

    CarryResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    r.xer          = ppc_state.spr[SPR::XER];
    r.cr           = ppc_state.cr;
    return r;
}

/** Emitted overflow reads the host OF where the interpreter compares sign
    bits by hand, and the two have to agree in XER, in CR and in every
    register, from a clean XER and from one with everything already up */
static void test_ov_subset() {
    load_ov_code();

    for (int pass = 0; pass < 2; pass++) {
        const uint32_t xer_in = pass ? (XER::SO | XER::OV | XER::CA) : 0;

        ppc_jit_disable();
        CarryResult interp = run_ov_code(xer_in);
        jit_check(interp.regs.pc == OV_END,
                  pass ? "the overflow program stopped where it should, with XER up"
                       : "the overflow program stopped where it should");

        ppc_jit_enable(JitBackend::threaded);
        CarryResult threaded = run_ov_code(xer_in);
        int where = -1;
        jit_check(same_carry(interp, threaded, &where),
                  pass ? "the threaded backend agrees on overflow, with XER up"
                       : "the threaded backend agrees on overflow");
        report_carry(interp, threaded, where, "threaded");

        ppc_jit_disable();
        ppc_jit_enable(JitBackend::automatic);
        CarryResult native = run_ov_code(xer_in);

        if (ppc_jit_native_compiles() == 0) {
            cout << "  (no emitter on this host, skipping the native comparison)" << endl;
            return;
        }

        where = -1;
        jit_check(same_carry(interp, native, &where),
                  pass ? "emitted code agrees on overflow, with XER up"
                       : "emitted code agrees on overflow");
        report_carry(interp, native, where, "emitted");
        jit_check(ppc_jit_threaded_compiles() == 0,
                  "the emitter took every overflow block instead of declining");
    }
}

/*  X forms and byte reversed ones: register indexed addressing, lwbrx both
    aligned and unaligned, stwbrx and sthbrx, and the update forms. Reuses
    the load data area at 0x3000 and the store area at 0x5000.  */
static const uint32_t xform_code[] = {
    0x3E800000, // addis  r20, r0, 0
    0x62943000, // ori    r20, r20, 0x3000
    0x3EA00000, // addis  r21, r0, 0
    0x62B55000, // ori    r21, r21, 0x5000
    0x3AC00004, // li     r22, 4
    0x7C74B02E, // lwzx   r3, r20, r22
    0x7C80A42C, // lwbrx  r4, 0, r20
    0x7CB4B42C, // lwbrx  r5, r20, r22
    0x7CC0A62C, // lhbrx  r6, 0, r20
    0x7CF4B0AE, // lbzx   r7, r20, r22
    0x7D14B22E, // lhzx   r8, r20, r22
    0x7D34B2AE, // lhax   r9, r20, r22
    0x7C60A92E, // stwx   r3, 0, r21
    0x7C95B52C, // stwbrx r4, r21, r22
    0x7CD5B32E, // sthx   r6, r21, r22      overwrites the middle of the stwbrx
    0x7CC0AF2C, // sthbrx r6, 0, r21        overwrites the middle of the stwx
    0x7CF5B1AE, // stbx   r7, r21, r22
    0x3AF40000, // addi   r23, r20, 0
    0x7DB7B06E, // lwzux  r13, r23, r22
    0x3B150000, // addi   r24, r21, 0
    0x7C98B16E, // stwux  r4, r24, r22
    0x3B200001, // li     r25, 1
    0x7F54CC2C, // lwbrx  r26, r20, r25     unaligned, slow path
    0x00000000, // illegal on purpose, the program stops here
};

constexpr uint32_t XFORM_BASE = 0x8200;
constexpr uint32_t XFORM_END  = XFORM_BASE + uint32_t(sizeof(xform_code) - 4);

static void load_xform_code() {
    for (size_t i = 0; i < sizeof(xform_code) / sizeof(xform_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, XFORM_BASE + uint32_t(i) * 4, xform_code[i]);
    }
    for (size_t i = 0; i < sizeof(load_data) / sizeof(load_data[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, LOAD_DATA + uint32_t(i) * 4, load_data[i]);
    }
}

static StoreResult run_xform_code() {
    for (int i = 0; i < STORE_WORDS; i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, STORE_DATA + uint32_t(i) * 4, 0xE5E5E5E5u);
    }
    for (int i = 0; i < 32; i++) {
        ppc_state.gpr[i] = 0x96960000u | uint32_t(i);
    }
    ppc_state.pc  = XFORM_BASE;
    g_icycles     = 0;
    g_icycles_max = 0;
    exec_flags    = 0;
    power_on      = true;

    ppc_exec_until(XFORM_END);

    StoreResult r;
    for (int i = 0; i < 32; i++) r.regs.gpr[i] = ppc_state.gpr[i];
    r.regs.pc      = ppc_state.pc;
    r.regs.retired = g_icycles;
    for (int i = 0; i < STORE_WORDS; i++) {
        r.mem[i] = mmu_read_vmem<uint32_t>(NO_OPCODE, STORE_DATA + uint32_t(i) * 4);
    }
    return r;
}

/** The X forms and the byte reversed ones, registers and memory, TLB cold
    and warm */
static void test_xform_subset() {
    load_xform_code();

    ppc_jit_disable();
    StoreResult interp = run_xform_code();
    jit_check(interp.regs.pc == XFORM_END, "the X form program stopped where it should");

    ppc_jit_enable(JitBackend::threaded);
    StoreResult threaded = run_xform_code();
    int where = -1;
    jit_check(same_store(interp, threaded, &where),
              "the threaded backend agrees on the X forms");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    StoreResult miss = run_xform_code();
    StoreResult hit  = run_xform_code();

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping the native comparison)" << endl;
        return;
    }

    where = -1;
    jit_check(same_store(interp, miss, &where),
              "emitted X forms agree on the first pass");
    if (where >= 100) {
        cout << "    memory +0x" << hex << (where - 100) * 4 << ": interpreter 0x"
             << interp.mem[where - 100] << ", emitted 0x" << miss.mem[where - 100]
             << dec << endl;
    } else if (where >= 0) {
        cout << "    r" << where << ": interpreter 0x" << hex << interp.regs.gpr[where]
             << ", emitted 0x" << miss.regs.gpr[where] << dec << endl;
    }

    where = -1;
    jit_check(same_store(interp, hit, &where), "and agree with the TLB already warm");
    jit_check(ppc_jit_threaded_compiles() == 0,
              "the emitter took every X form block instead of declining");
}

/*  Six ordinary instructions precede an MMIO load, and the loaded value is
    then written back to the device. The read must observe six retired guest
    instructions and the following write must observe seven. */
static const uint32_t mmio_cycle_code[] = {
    0x38600000, // li    r3, 0
    0x38630001, // addi  r3, r3, 1
    0x38630001, // addi  r3, r3, 1
    0x38630001, // addi  r3, r3, 1
    0x3D40F000, // lis   r10, 0xF000
    0x614A0000, // ori   r10, r10, 0
    0x808A0000, // lwz   r4, 0(r10)
    0x908A0004, // stw   r4, 4(r10)
    0x00000000, // illegal on purpose, marks the end
};

constexpr uint32_t MMIO_CYCLE_BASE = 0xE000;
constexpr uint32_t MMIO_CYCLE_END  = MMIO_CYCLE_BASE + uint32_t(sizeof(mmio_cycle_code) - 4);
constexpr uint32_t MMIO_CYCLE_ADDR = 0xF0000000;

struct MmioCycleResult {
    uint32_t loaded;
    uint32_t written;
    uint32_t pc;
    uint64_t read_cycle;
    uint64_t write_cycle;
    uint64_t retired;
};

static void load_mmio_cycle_code() {
    for (size_t i = 0; i < sizeof(mmio_cycle_code) / sizeof(mmio_cycle_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, MMIO_CYCLE_BASE + uint32_t(i) * 4,
                                 mmio_cycle_code[i]);
    }
}

static MmioCycleResult run_mmio_cycle_code(JitCycleMmio& device) {
    device.reset();
    ppc_state.pc      = MMIO_CYCLE_BASE;
    ppc_state.gpr[3]  = 0;
    ppc_state.gpr[4]  = 0;
    ppc_state.gpr[10] = 0;
    g_icycles         = 0;
    g_icycles_max     = UINT64_MAX;
    exec_timer        = false;
    exec_flags        = 0;
    power_on          = true;

    ppc_exec_until(MMIO_CYCLE_END);

    return {ppc_state.gpr[4], device.write_value, ppc_state.pc,
            device.read_cycle, device.write_cycle, g_icycles};
}

static bool same_mmio_cycle(const MmioCycleResult& a, const MmioCycleResult& b) {
    return a.loaded == b.loaded && a.written == b.written && a.pc == b.pc &&
           a.read_cycle == b.read_cycle && a.write_cycle == b.write_cycle &&
           a.retired == b.retired;
}

static void test_mmio_cycle_visibility(MPC106* host_bridge) {
    static JitCycleMmio device;
    if (!host_bridge->add_mmio_region(MMIO_CYCLE_ADDR, 0x1000, &device)) {
        jit_check(false, "the MMIO cycle test region was registered");
        return;
    }
    load_mmio_cycle_code();

    ppc_jit_disable();
    const MmioCycleResult interp = run_mmio_cycle_code(device);
    jit_check(interp.read_cycle == 6 && interp.loaded == 6,
              "the interpreter exposes prior cycles to an MMIO read");
    jit_check(interp.write_cycle == 7 && interp.written == 6,
              "the interpreter exposes prior cycles to an MMIO write");
    jit_check(interp.retired == 8 && interp.pc == MMIO_CYCLE_END,
              "the MMIO program retired exactly eight instructions");

    ppc_jit_enable(JitBackend::threaded);
    const MmioCycleResult threaded = run_mmio_cycle_code(device);
    jit_check(same_mmio_cycle(interp, threaded),
              "the threaded backend exposes the same cycles to MMIO");

    ppc_jit_disable();
    ppc_jit_enable(JitBackend::automatic);
    const MmioCycleResult native = run_mmio_cycle_code(device);
    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping native MMIO timing)" << endl;
    } else {
        jit_check(same_mmio_cycle(interp, native),
                  "emitted code exposes the same cycles to MMIO");

        const MmioCycleResult cached = run_mmio_cycle_code(device);
        jit_check(same_mmio_cycle(interp, cached),
                  "a cached native block keeps MMIO cycle visibility exact");
    }
}

/* A one-block counted loop. Its taken bdnz targets the block itself, which is
   the smallest shape that can prove the resolver bound a same-page chain. */
static const uint32_t chain_code[] = {
    0x38630001, // addi r3, r3, 1
    0x4200FFFC, // bdnz -4
    0x00000000, // test-only stop helper installed at this decode slot
};

constexpr uint32_t CHAIN_BASE       = 0xB000;
constexpr uint32_t CHAIN_END        = CHAIN_BASE + 8;
constexpr uint32_t CHAIN_ITERATIONS = 64;

static void stop_chain_run(uint32_t) {
    power_on = false;
}

static void load_chain_code() {
    for (size_t i = 0; i < sizeof(chain_code) / sizeof(chain_code[0]); i++) {
        mmu_write_vmem<uint32_t>(NO_OPCODE, CHAIN_BASE + uint32_t(i) * 4, chain_code[i]);
    }
}

static uint32_t run_chain_code(bool until) {
    ppc_state.pc            = CHAIN_BASE;
    ppc_state.gpr[3]        = 0;
    ppc_state.spr[SPR::CTR] = CHAIN_ITERATIONS;
    g_icycles               = 0;
    g_icycles_max           = UINT64_MAX;
    exec_timer              = false;
    exec_flags              = 0;
    power_on                = true;

    if (until) {
        ppc_exec_until(CHAIN_END);
    } else {
        ppc_exec();
    }
    return ppc_state.gpr[3];
}

/* The same counted-loop shape split across two pages. The forward b can be
   followed into the second page, while the backward bdnz has to use the
   guarded virtual-address chain slot to return to the first one. */
constexpr uint32_t VA_CHAIN_A   = 0xC000;
constexpr uint32_t VA_CHAIN_B   = 0xD000;
constexpr uint32_t VA_CHAIN_END = VA_CHAIN_B + 8;

static void load_va_chain_code() {
    mmu_write_vmem<uint32_t>(NO_OPCODE, VA_CHAIN_A,     0x38630001); // addi r3,r3,1
    mmu_write_vmem<uint32_t>(NO_OPCODE, VA_CHAIN_A + 4, 0x48000FFC); // b VA_CHAIN_B
    mmu_write_vmem<uint32_t>(NO_OPCODE, VA_CHAIN_B,     0x38630001); // addi r3,r3,1
    mmu_write_vmem<uint32_t>(NO_OPCODE, VA_CHAIN_B + 4, 0x4200EFFC); // bdnz VA_CHAIN_A
    mmu_write_vmem<uint32_t>(NO_OPCODE, VA_CHAIN_END,   0x00000000);
}

static uint32_t run_va_chain_code(bool until) {
    ppc_state.pc              = VA_CHAIN_A;
    ppc_state.gpr[3]          = 0;
    ppc_state.spr[SPR::CTR]   = CHAIN_ITERATIONS;
    g_icycles                 = 0;
    g_icycles_max             = UINT64_MAX;
    exec_timer                = false;
    exec_flags                = 0;
    power_on                  = true;

    if (until) {
        ppc_exec_until(VA_CHAIN_END);
    } else {
        ppc_exec();
    }
    return ppc_state.gpr[3];
}

/* One bclr site alternating between two target pages. This is the workload
   the two ChainVaSlot ways exist for: neither prediction should evict the
   other after the first pair of visits. */
constexpr uint32_t ALT_CHAIN_A        = 0x8000;
constexpr uint32_t ALT_CHAIN_B        = 0x9000;
constexpr uint32_t ALT_CHAIN_DISPATCH = 0xE000;
constexpr uint32_t ALT_CHAIN_END      = ALT_CHAIN_A + 8;

static void load_alt_chain_code() {
    // r4 toggles 0/1, which selects 0x8000/0x9000 without a control-flow
    // split before the one bclr whose two ways are under test.
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_DISPATCH,      0x68840001); // xori r4,r4,1
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_DISPATCH + 4,  0x1CA41000); // mulli r5,r4,0x1000
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_DISPATCH + 8,  0x60A58000); // ori r5,r5,0x8000
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_DISPATCH + 12, 0x7CA803A6); // mtlr r5
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_DISPATCH + 16, 0x4E800020); // blr

    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_A,     0x38630001); // addi r3,r3,1
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_A + 4, 0x42005FFC); // bdnz dispatch
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_A + 8, 0x00000000);
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_B,     0x3863000A); // addi r3,r3,10
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_B + 4, 0x42004FFC); // bdnz dispatch
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_B + 8, 0x00000000);
}

static uint32_t run_alt_chain_code(bool until) {
    ppc_state.pc              = ALT_CHAIN_DISPATCH;
    ppc_state.gpr[3]          = 0;
    ppc_state.gpr[4]          = 0;
    ppc_state.spr[SPR::CTR]   = CHAIN_ITERATIONS;
    ppc_state.spr[SPR::LR]    = 0;
    g_icycles                 = 0;
    g_icycles_max             = UINT64_MAX;
    exec_timer                = false;
    exec_flags                = 0;
    power_on                  = true;

    if (until) {
        ppc_exec_until(ALT_CHAIN_END);
    } else {
        ppc_exec();
    }
    return ppc_state.gpr[3];
}

static void test_native_chaining() {
    ppc_jit_disable();
    if (!ppc_jit_enable(JitBackend::automatic)) {
        cout << "  (no emitter on this host, skipping native chaining)" << endl;
        return;
    }
    ppc_jit_flush();

    // Opcode zero is normally illegal. Making just this table entry a
    // controlled stop lets ppc_exec exercise real chaining without entering
    // an exception vector that the test's small RAM image does not map.
    PPCOpcode saved_zero = ppc_opcode_grabber[0];
    ppc_opcode_grabber[0] = stop_chain_run;
    load_chain_code();

    const uint32_t first_result = run_chain_code(false);
    jit_check(first_result == CHAIN_ITERATIONS,
              "a ppc_exec run completed the chained counted loop");

    if (ppc_jit_native_compiles() == 0) {
        cout << "  (no emitter on this host, skipping native chaining)" << endl;
        ppc_opcode_grabber[0] = saved_zero;
        ppc_jit_flush();
        return;
    }

    jit_check(ppc_jit_bound_chains() > 0,
              "the loop left live same-page chain bindings");

    const uint32_t until_result = run_chain_code(true);
    jit_check(until_result == CHAIN_ITERATIONS,
              "the same loop remains correct under ppc_exec_until");
    jit_check(g_icycles == CHAIN_ITERATIONS * 2,
              "same-page chaining retires each guest instruction exactly once");
    jit_check(ppc_jit_bound_chains() == 0,
              "an until run unbound every previously chained entry");

    // The until resolver remembers its observation without turning it into a
    // direct jump. That weak cache lives on the same incoming registry list,
    // so killing the target must clear it without underflowing the bound count
    // or leaving a dangling JitBlock pointer in the source slot.
    const unsigned cached_blocks = ppc_jit_num_blocks();
    mmu_write_vmem<uint32_t>(NO_OPCODE, CHAIN_BASE, 0x60000000);
    jit_check(ppc_jit_bound_chains() == 0,
              "invalidating an observed target keeps the bound count at zero");
    jit_check(ppc_jit_num_blocks() + 1 == cached_blocks,
              "invalidating an observed target removed exactly its block");
    load_chain_code();

    const uint32_t rebound_result = run_chain_code(false);
    jit_check(rebound_result == CHAIN_ITERATIONS,
              "the loop remains correct after returning to ppc_exec");
    jit_check(ppc_jit_bound_chains() > 0,
              "an ordinary run rebound the chain lazily");

    // A write into the loop invalidates its block. This exercises both sides
    // of the registry: the self-chain is incoming to that block, while its
    // fall-through chain is an outgoing entry aimed at the stop block.
    const unsigned blocks_before_invalidation = ppc_jit_num_blocks();
    mmu_write_vmem<uint32_t>(NO_OPCODE, CHAIN_BASE, 0x60000000);
    jit_check(ppc_jit_bound_chains() == 0,
              "invalidating a chain source detached incoming and outgoing slots");
    jit_check(ppc_jit_num_blocks() + 1 == blocks_before_invalidation,
              "invalidating the loop removed exactly its translated block");

    load_chain_code();
    const uint32_t restored_result = run_chain_code(false);
    jit_check(restored_result == CHAIN_ITERATIONS,
              "the restored loop still computes the expected result");
    jit_check(ppc_jit_bound_chains() > 0,
              "the restored block acquired a fresh chain binding");

    // Repeat the lifecycle through ChainVaSlot. Its two guarded ways keep
    // observed targets as well as direct bindings, so an until loop can avoid
    // translating the same cross-page destination on every turn without
    // ever skipping the goal observation.
    ppc_jit_flush();
    load_va_chain_code();
    const uint32_t va_result = run_va_chain_code(false);
    jit_check(va_result == CHAIN_ITERATIONS * 2,
              "a cross-page ppc_exec loop completed through a virtual chain");
    jit_check(ppc_jit_bound_chains() > 0,
              "the cross-page loop left a guarded virtual chain binding");

    const uint32_t va_until_result = run_va_chain_code(true);
    jit_check(va_until_result == CHAIN_ITERATIONS * 2,
              "the cross-page loop remains correct under ppc_exec_until");
    jit_check(g_icycles == CHAIN_ITERATIONS * 4,
              "virtual chaining retires each guest instruction exactly once");
    jit_check(ppc_jit_bound_chains() == 0,
              "the virtual until run retained no direct bindings");

    const unsigned va_cached_blocks = ppc_jit_num_blocks();
    mmu_write_vmem<uint32_t>(NO_OPCODE, VA_CHAIN_A, 0x60000000);
    jit_check(ppc_jit_bound_chains() == 0,
              "invalidating an observed virtual target keeps the count at zero");
    jit_check(ppc_jit_num_blocks() + 1 == va_cached_blocks,
              "invalidating an observed virtual target removed exactly its block");

    load_va_chain_code();
    const uint32_t va_cached_result = run_va_chain_code(true);
    jit_check(va_cached_result == CHAIN_ITERATIONS * 2,
              "a restored virtual chain repopulated its observation cache");
    const uint32_t va_rebound_result = run_va_chain_code(false);
    jit_check(va_rebound_result == CHAIN_ITERATIONS * 2,
              "a virtual observation promoted without changing guest results");
    jit_check(ppc_jit_bound_chains() > 0,
              "an ordinary run promoted the virtual observation to a binding");

    ppc_jit_flush();
    load_alt_chain_code();
    const uint32_t alt_until_result = run_alt_chain_code(true);
    jit_check(alt_until_result == (CHAIN_ITERATIONS / 2) * 11,
              "two observed virtual ways alternate without changing targets");
    jit_check(g_icycles == CHAIN_ITERATIONS * 7,
              "alternating virtual ways do not leak a retired count");
    jit_check(ppc_jit_bound_chains() == 0,
              "two observed virtual ways still count as unbound");

    const unsigned alt_cached_blocks = ppc_jit_num_blocks();
    mmu_write_vmem<uint32_t>(NO_OPCODE, ALT_CHAIN_A, 0x60000000);
    jit_check(ppc_jit_bound_chains() == 0,
              "invalidating one observed way leaves no phantom binding");
    jit_check(ppc_jit_num_blocks() + 1 == alt_cached_blocks,
              "invalidating one observed way removes only its target block");

    load_alt_chain_code();
    const uint32_t alt_restored_result = run_alt_chain_code(true);
    jit_check(alt_restored_result == (CHAIN_ITERATIONS / 2) * 11,
              "the alternating observation cache repopulates after invalidation");
    const uint32_t alt_bound_result = run_alt_chain_code(false);
    jit_check(alt_bound_result == (CHAIN_ITERATIONS / 2) * 11,
              "both observed ways promote without changing alternation");
    jit_check(ppc_jit_bound_chains() > 0,
              "the alternating site promoted its virtual ways to bindings");

    ppc_opcode_grabber[0] = saved_zero;
    ppc_jit_flush();
}

int test_jit() {
    jit_tested = 0;
    jit_failed = 0;

    // These short programs are emitter coverage tests, so compile on their
    // first entry except in the dedicated heat-gate test below.
    set_test_heat(false);

    MPC106* host_bridge = new MPC106;

    if (!host_bridge->add_ram_region(0, 0x10000)) {
        cout << "  Failed: could not create a RAM region for the JIT tests" << endl;
        delete host_bridge;
        return 1;
    }

    ppc_cpu_init(host_bridge, PPC_VER::MPC750, false, 16705000);

#if defined(DPPC_JIT_X86_64)
    test_x64_encoding();
#endif

    // MSR[DR] is clear coming out of reset, so effective addresses are physical
    load_test_code();

    // the reference runs, still on the interpreter
    RunResult interp     = run_test_code();
    RunResult interp_mid = run_test_code(MID_BLOCK);

    // Cache and invalidation tests need real blocks. Probe automatic mode
    // first; the minimal AArch64 bring-up and hosts without an emitter use
    // explicit threaded blocks for those machinery tests, while automatic
    // interpreter fallback is verified separately below.
    bool have_emitter = false;
    if (ppc_jit_enable(JitBackend::automatic)) {
        run_test_code();
        have_emitter = ppc_jit_native_compiles() != 0;
        ppc_jit_disable();
    }
    if (!ppc_jit_enable(have_emitter ? JitBackend::automatic : JitBackend::threaded)) {
        cout << "  Failed: no JIT test backend came up" << endl;
        return 1;
    }
    jit_check(ppc_jit_is_enabled(), "the JIT reports itself enabled");
    cout << "  backend: " << ppc_jit_backend_name() << endl;

    test_parity(interp);
    test_icbi_invalidation(interp);
    test_store_invalidation(interp);
    test_mode_is_part_of_the_key(interp);
    test_mid_block_goal(interp_mid);
    test_native_matches_threaded(interp);
    test_interpreter_heat_gate();
    test_constant_fold_ir();
    test_memory_gpr_cache_ir();
    test_alu_subset();
    test_load_subset();
    test_store_subset();
    test_cr_subset();
    test_branch_subset();
    test_superblock_subset();
    test_superblock_budget_edge();
    test_cross_page_call();
    test_leaf_lr_write();
    test_leaf_called_twice();
    test_cr_fusion();
    test_cr_fusion_matrix();
    test_carry_subset();
    test_ov_subset();
    test_xform_subset();
    test_mmio_cycle_visibility(host_bridge);
    test_native_chaining();

    ppc_jit_disable();
    jit_check(!ppc_jit_is_enabled(), "the JIT reports itself disabled again");
    jit_check(ppc_jit_num_blocks() == 0, "turning it off dropped every block");

    RunResult back_on_interp = run_test_code();
    jit_check(same_run(interp, back_on_interp), "the interpreter still runs it the same");

    cout << "--> Tested: " << dec << jit_tested << endl;
    cout << "--> Failed: " << dec << jit_failed << endl;

    return jit_failed;
}
