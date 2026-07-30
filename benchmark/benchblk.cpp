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

/** @file What a block boundary costs, as a function of block length.

    bench1 answers a different question. Its inner loop is long enough that
    whatever the executor spends between blocks is spread thin, and a real
    operating system does not look like that: the boot profile of Mac OS X
    puts a branch every 4.3 guest instructions, so the boundary is paid four
    times more often than bench1 suggests.

    This runs the same loop at several block lengths and divides by the
    instructions retired, which separates the two costs. The body is one
    addi repeated, so what changes between lengths is only how often the
    boundary is crossed. The gap between the shortest and the longest run is
    the per block cost, and driving it down is the whole point of the shared
    frame and of chaining blocks together.

    The loop closes with bdnz, which is a taken backward branch to the top of
    the same block. That is deliberate: it is both the commonest shape in a
    real workload and the one a chained block can serve without leaving
    generated code at all.

    Set DPPC_JIT=1 to measure the emitter, DPPC_JIT=threaded for the portable
    backend, and leave it unset for the interpreter.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cpu/ppc/ppcemu.h"
#include "cpu/ppc/ppcjit.h"
#include "cpu/ppc/ppcmmu.h"
#include "devices/memctrl/mpc106.h"
#include <thirdparty/loguru/loguru.hpp>

#if defined(PPC_BENCHMARKS)
void ppc_exception_handler(Except_Type exception_type, uint32_t srr1_bits) {
    power_on = false;
    power_off_reason = po_benchmark_exception;
}
#endif

namespace {

constexpr uint32_t CODE_BASE = 0x1000;

/** Block lengths to sweep. 4 is what a real workload averages, 64 is the
    translator's own ceiling, and the ones between show the shape */
const uint32_t lengths[] = {2, 4, 8, 16, 33, 64};

constexpr uint32_t samples    = 30;
constexpr uint64_t iterations = 200000;

/** addi r3, r3, 1 */
constexpr uint32_t ADDI_R3 = 0x38630001;

/** bc 16, 0, disp: decrement CTR and branch back while it is not zero */
uint32_t bdnz(int32_t disp) {
    return 0x42000000UL | (uint32_t(disp) & 0xFFFCUL);
}

/** Writes a loop of `len` guest instructions, all but the last an addi and
    the last a bdnz back to the top. One block per iteration, because a
    branch is what ends a block */
void write_loop(uint32_t len) {
    for (uint32_t i = 0; i + 1 < len; i++) {
        mmu_write_vmem<uint32_t>(0, CODE_BASE + i * 4, ADDI_R3);
    }
    const int32_t back = -int32_t((len - 1) * 4);
    mmu_write_vmem<uint32_t>(0, CODE_BASE + (len - 1) * 4, bdnz(back));
}

/** Best of `samples` runs, in nanoseconds, with the timing overhead removed */
uint64_t time_loop(uint32_t len, uint64_t overhead) {
    const uint32_t goal = CODE_BASE + len * 4;

    uint64_t best = uint64_t(-1);
    for (uint32_t s = 0; s < samples; s++) {
        ppc_state.pc          = CODE_BASE;
        ppc_state.gpr[3]      = 0;
        ppc_state.spr[SPR::CTR] = uint32_t(iterations);
        power_on              = true;

        auto start = std::chrono::steady_clock::now();
        ppc_exec_until(goal);
        auto end   = std::chrono::steady_clock::now();

        const uint64_t ns = uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ns < best) {
            best = ns;
        }

        if (ppc_state.gpr[3] != uint32_t(iterations * (len - 1))) {
            LOG_F(ERROR, "block length %u: r3 is 0x%08X, expected 0x%08X",
                  len, ppc_state.gpr[3], uint32_t(iterations * (len - 1)));
            return 0;
        }
    }
    return best > overhead ? best - overhead : 0;
}

} // namespace

int main(int argc, char** argv) {
    loguru::g_preamble_date   = false;
    loguru::g_preamble_time   = false;
    loguru::g_preamble_thread = false;
    loguru::g_stderr_verbosity = 0;
    loguru::init(argc, argv);

    MPC106* grackle = new MPC106;
    if (!grackle->add_ram_region(0, 0x10000)) {
        LOG_F(ERROR, "Could not create RAM region");
        delete grackle;
        return -1;
    }

    ppc_cpu_init(grackle, PPC_VER::MPC750, false, 16705000);

    uint64_t overhead = uint64_t(-1);
    for (uint32_t s = 0; s < samples; s++) {
        auto a = std::chrono::steady_clock::now();
        auto b = std::chrono::steady_clock::now();
        const uint64_t ns = uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
        if (ns < overhead) {
            overhead = ns;
        }
    }

    printf("%6s %14s %14s %14s\n", "insns", "total ns", "ns/insn", "ns/block");
    for (uint32_t len : lengths) {
        write_loop(len);

        // the guest code just changed underneath any translation of it
        ppc_jit_flush();

        const uint64_t ns = time_loop(len, overhead);
        if (!ns) {
            continue;
        }
        const double per_insn  = double(ns) / double(iterations * len);
        const double per_block = double(ns) / double(iterations);
        printf("%6u %14llu %14.4f %14.4f\n", len,
               (unsigned long long)ns, per_insn, per_block);
        fflush(stdout);
    }

    delete grackle;
    return 0;
}
