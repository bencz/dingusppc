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

/** @file x86-64 host processor capability discovery. */

#include "../host/host_cpu_features.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cstdint>

namespace dppc_jit {

namespace {

constexpr uint32_t CPUID_1_ECX_MOVBE = 1u << 22;

HostCpuFeatures detect_host_cpu_features() {
    HostCpuFeatures features;

#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] >= 1) {
        __cpuidex(regs, 1, 0);
        features.x86_movbe = (uint32_t(regs[2]) & CPUID_1_ECX_MOVBE) != 0;
    }
#else
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        features.x86_movbe = (ecx & CPUID_1_ECX_MOVBE) != 0;
    }
#endif

    return features;
}

} // namespace

const HostCpuFeatures& host_cpu_features() {
    static const HostCpuFeatures features = detect_host_cpu_features();
    return features;
}

} // namespace dppc_jit
