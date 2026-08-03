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
    boundary is crossed. Each length is measured with ppc_exec_until, which
    keeps bindings disabled and exposes resolver cost, and with ppc_exec,
    which lets the backward edge bind and stay inside generated code.

    The first suite closes with bdnz to the top of the same page. The second
    splits each turn across two pages, so its backward edge exercises the
    guarded virtual-address resolver used by returns, bcctr and cross-page
    branches. Together they keep same-page and mapping-sensitive chaining
    visible as separate numbers.

    Set DPPC_JIT=1 to measure the emitter, DPPC_JIT=threaded for the portable
    backend, and leave it unset for the interpreter. DPPC_BENCH_SAMPLES and
    DPPC_BENCH_ITERATIONS override their defaults (30 and 200000) for quicker
    diagnostic sweeps.
 */

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

constexpr uint32_t CODE_BASE  = 0x1000;
constexpr uint32_t CROSS_BASE = 0x2000;

enum class RunMode {
    unchained,
    chained,
};

enum class LoopShape {
    same_page,
    cross_page,
};

/** Block lengths to sweep. 4 is what a real workload averages, 64 is the
    translator's own ceiling, and the ones between show the shape */
const uint32_t lengths[] = {2, 4, 8, 16, 33, 64};
const uint32_t cross_lengths[] = {2, 4, 8, 16, 32};

uint32_t samples    = 30;
uint32_t iterations = 200000;

bool read_env_uint32(const char* name, uint32_t& value) {
    const char* text = std::getenv(name);
    if (!text || !*text) {
        return true;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || *end || parsed == 0 ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        LOG_F(ERROR, "%s must be an integer from 1 through %u", name,
              std::numeric_limits<uint32_t>::max());
        return false;
    }

    value = uint32_t(parsed);
    return true;
}

/** addi r3, r3, 1 */
constexpr uint32_t ADDI_R3 = 0x38630001;

/** bc 16, 0, disp: decrement CTR and branch back while it is not zero */
uint32_t bdnz(int32_t disp) {
    return 0x42000000UL | (uint32_t(disp) & 0xFFFCUL);
}

/** b disp: an unconditional relative branch. */
uint32_t branch(int32_t disp) {
    return 0x48000000UL | (uint32_t(disp) & 0x03FFFFFCUL);
}

/** Writes a loop of `len` guest instructions, all but the last an addi and
    the last a bdnz back to the top. One block per iteration, because a
    branch is what ends a block. The illegal word just beyond it stops a
    ppc_exec run through the benchmark's exception handler. */
void write_loop(uint32_t len) {
    for (uint32_t i = 0; i + 1 < len; i++) {
        mmu_write_vmem<uint32_t>(0, CODE_BASE + i * 4, ADDI_R3);
    }
    const int32_t back = -int32_t((len - 1) * 4);
    mmu_write_vmem<uint32_t>(0, CODE_BASE + (len - 1) * 4, bdnz(back));
    mmu_write_vmem<uint32_t>(0, CODE_BASE + len * 4, 0);
}

/** Writes one logical loop split across two pages.

    The first half ends in a forward direct branch. The translator can follow
    that into the second page behind its instruction-translation guard, so the
    backward bdnz is a virtual-address exit back into the first page. Even if
    the block-size limit keeps a long pair separate, both page crossings still
    exercise ChainVaSlot rather than the same-page ChainSlot measured above. */
void write_cross_loop(uint32_t len) {
    for (uint32_t i = 0; i + 1 < len; i++) {
        mmu_write_vmem<uint32_t>(0, CODE_BASE + i * 4, ADDI_R3);
        mmu_write_vmem<uint32_t>(0, CROSS_BASE + i * 4, ADDI_R3);
    }
    mmu_write_vmem<uint32_t>(
        0, CODE_BASE + (len - 1) * 4,
        branch(int32_t(CROSS_BASE) - int32_t(CODE_BASE + (len - 1) * 4)));
    mmu_write_vmem<uint32_t>(
        0, CROSS_BASE + (len - 1) * 4,
        bdnz(int32_t(CODE_BASE) - int32_t(CROSS_BASE + (len - 1) * 4)));
    mmu_write_vmem<uint32_t>(0, CROSS_BASE + len * 4, 0);
}

/** Best of `samples` runs, in nanoseconds, with the timing overhead removed */
uint64_t time_loop(uint32_t len, uint64_t overhead, RunMode mode,
                   LoopShape shape) {
    const uint32_t goal = shape == LoopShape::same_page
        ? CODE_BASE + len * 4
        : CROSS_BASE + len * 4;
    const uint32_t halves = shape == LoopShape::same_page ? 1 : 2;

    uint64_t best = uint64_t(-1);
    for (uint32_t s = 0; s < samples; s++) {
        ppc_state.pc          = CODE_BASE;
        ppc_state.gpr[3]      = 0;
        ppc_state.spr[SPR::CTR] = uint32_t(iterations);
        power_on              = true;

        auto start = std::chrono::steady_clock::now();
        if (mode == RunMode::chained) {
            ppc_exec();
        } else {
            ppc_exec_until(goal);
        }
        auto end   = std::chrono::steady_clock::now();

        const uint64_t ns = uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ns < best) {
            best = ns;
        }

        const uint32_t expected = uint32_t(iterations * (len - 1) * halves);
        if (ppc_state.gpr[3] != expected) {
            LOG_F(ERROR, "block length %u: r3 is 0x%08X, expected 0x%08X",
                  len, ppc_state.gpr[3], expected);
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

    if (!read_env_uint32("DPPC_BENCH_SAMPLES", samples) ||
        !read_env_uint32("DPPC_BENCH_ITERATIONS", iterations)) {
        return -1;
    }
    printf("samples=%u iterations=%u\n", samples, iterations);

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

    for (LoopShape shape : {LoopShape::same_page, LoopShape::cross_page}) {
        const uint32_t* shape_lengths = shape == LoopShape::same_page
            ? lengths : cross_lengths;
        const size_t shape_count = shape == LoopShape::same_page
            ? sizeof(lengths) / sizeof(lengths[0])
            : sizeof(cross_lengths) / sizeof(cross_lengths[0]);

        for (RunMode mode : {RunMode::unchained, RunMode::chained}) {
            printf("\n%s %s (%s)\n",
                   shape == LoopShape::same_page ? "same-page" : "cross-page VA",
                   mode == RunMode::chained ? "chained" : "unchained",
                   mode == RunMode::chained ? "ppc_exec" : "ppc_exec_until");
            printf("%6s %14s %14s %14s %12s\n",
                   "insns", "total ns", "ns/insn", "ns/turn", "MIPS");

            for (size_t index = 0; index < shape_count; index++) {
                const uint32_t len = shape_lengths[index];
                const uint32_t halves = shape == LoopShape::same_page ? 1 : 2;
                if (shape == LoopShape::same_page) {
                    write_loop(len);
                } else {
                    write_cross_loop(len);
                }

                // The guest code just changed underneath any translation of it.
                // Flushing per mode also keeps one measurement from donating a
                // warm binding to the other.
                ppc_jit_flush();

                const uint64_t ns = time_loop(len, overhead, mode, shape);
                if (!ns) {
                    continue;
                }
                const uint32_t insns_per_turn = len * halves;
                const double per_insn =
                    double(ns) / double(iterations * insns_per_turn);
                const double per_turn = double(ns) / double(iterations);
                const double mips     = 1000.0 / per_insn;
                printf("%6u %14llu %14.4f %14.4f %12.1f\n",
                       insns_per_turn, (unsigned long long)ns,
                       per_insn, per_turn, mips);
                fflush(stdout);
            }
        }
    }

    delete grackle;
    return 0;
}
