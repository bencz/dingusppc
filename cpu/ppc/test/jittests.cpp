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

#include "../ppccodecache.h"
#include "../ppcemu.h"
#include "../ppcjit.h"
#include "../ppcmmu.h"
#include "devices/memctrl/mpc106.h"

#include <iostream>

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

/** A store into a page holding blocks takes them down through the PAGE_CODE
    path, without the guest having to run icbi at all */
static void test_store_invalidation(const RunResult& interp) {
    jit_check(ppc_jit_num_blocks() == 2, "starting from two cached blocks");

    // somewhere on the code page but past the code itself
    mmu_write_vmem<uint32_t>(NO_OPCODE, CODE_BASE + 0x800, 0x12345678);
    jit_check(ppc_jit_num_blocks() == 0, "a store to the code page dropped the blocks");

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
        jit_check(false, "the automatic backend came up");
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
    0x38A5FFFF, // addi   r5, r5, -1
    0x7CA52A14, // add    r5, r5, r5
    0x7CB02B78, // mr     r16, r5
    0x7E232050, // subf   r17, r3, r4
    0x50B2400E, // rlwimi r18, r5, 8, 0, 7
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

int test_jit() {
    jit_tested = 0;
    jit_failed = 0;

    MPC106* host_bridge = new MPC106;

    if (!host_bridge->add_ram_region(0, 0x10000)) {
        cout << "  Failed: could not create a RAM region for the JIT tests" << endl;
        delete host_bridge;
        return 1;
    }

    ppc_cpu_init(host_bridge, PPC_VER::MPC750, false, 16705000);

    // MSR[DR] is clear coming out of reset, so effective addresses are physical
    load_test_code();

    // the reference runs, still on the interpreter
    RunResult interp     = run_test_code();
    RunResult interp_mid = run_test_code(MID_BLOCK);

    if (!ppc_jit_enable()) {
        cout << "  Failed: the JIT refused to come up" << endl;
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
    test_alu_subset();
    test_load_subset();
    test_store_subset();
    test_cr_subset();
    test_branch_subset();
    test_carry_subset();
    test_xform_subset();

    ppc_jit_disable();
    jit_check(!ppc_jit_is_enabled(), "the JIT reports itself disabled again");
    jit_check(ppc_jit_num_blocks() == 0, "turning it off dropped every block");

    RunResult back_on_interp = run_test_code();
    jit_check(same_run(interp, back_on_interp), "the interpreter still runs it the same");

    cout << "--> Tested: " << dec << jit_tested << endl;
    cout << "--> Failed: " << dec << jit_failed << endl;

    return jit_failed;
}
