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

/** @file Tests for cycle accounting.

    The interpreter accounts one instruction at a time. A block executor will
    account for a whole block at once, so what matters is that batching never
    fires a timer early and never delays one by more than the batch length.
 */

#include "../ppcemu.h"
#include "devices/memctrl/mpc106.h"

#include <core/timermanager.h>
#include <iostream>
#include <string>

using namespace std;

static int      cy_tested;
static int      cy_failed;
static bool     timer_fired;
static uint64_t timer_fire_icycles;

static void cy_check(bool passed, const string& what) {
    cy_tested++;
    if (!passed) {
        cout << "  Failed: " << what << endl;
        cy_failed++;
    }
}

static void on_test_timer() {
    timer_fired        = true;
    timer_fire_icycles = g_icycles;
}

/** Arms a timer due after `deadline` instructions, then accounts for cycles
    `chunk` at a time and reports where it went off */
static uint64_t fire_point(uint64_t deadline, uint64_t chunk) {
    g_icycles          = 0;
    g_icycles_max      = 0;
    exec_timer         = false;
    timer_fired        = false;
    timer_fire_icycles = 0;

    uint32_t id = TimerManager::get_instance()->add_oneshot_timer(
        deadline << icnt_factor, on_test_timer);

    for (uint64_t i = 0; i < deadline * 4 && !timer_fired; i += chunk) {
        ppc_account_cycles(chunk);
    }

    if (!timer_fired) {
        TimerManager::get_instance()->cancel_timer(id);
    }
    return timer_fire_icycles;
}

static void test_deadline() {
    constexpr uint64_t deadline = 1000;

    g_icycles     = 0;
    g_icycles_max = 0;
    exec_timer    = false;

    uint32_t id = TimerManager::get_instance()->add_oneshot_timer(
        deadline << icnt_factor, on_test_timer);
    ppc_account_cycles(1);
    cy_check(g_icycles_max > g_icycles && g_icycles_max <= deadline + 2,
             "the deadline lands on the next timer");
    TimerManager::get_instance()->cancel_timer(id);

    g_icycles     = 0;
    g_icycles_max = 0;
    exec_timer    = true;
    ppc_account_cycles(1);
    cy_check(g_icycles_max == 1 + 25000, "with nothing pending the deadline is a fixed slice");
}

static void test_batching() {
    constexpr uint64_t deadline = 1000;

    uint64_t one_at_a_time = fire_point(deadline, 1);
    cy_check(one_at_a_time >= deadline,
             "accounting one instruction at a time never fires a timer early");

    for (uint64_t chunk : {4ULL, 64ULL, 1024ULL}) {
        const string tail = " with a batch of " + to_string(chunk);
        uint64_t batched  = fire_point(deadline, chunk);

        cy_check(batched != 0, "the timer still fires" + tail);
        cy_check(batched >= one_at_a_time, "no timer fires early" + tail);
        cy_check(batched <= one_at_a_time + chunk,
                 "a timer is late by at most the batch length" + tail);
    }
}

static void test_realtime_keeps_virtual_time() {
    const uint64_t saved_icycles = g_icycles;
    g_icycles = 0x123456;
    ppc_set_realtime(true);
    cy_check(get_virt_time_ns() == (g_icycles << icnt_factor),
             "real-time pacing leaves guest virtual time cycle-derived");
    ppc_set_realtime(false);
    g_icycles = saved_icycles;
}

int test_cycle_accounting() {
    cy_tested = 0;
    cy_failed = 0;

    MPC106* host_bridge = new MPC106;
    host_bridge->add_ram_region(0, 0x10000);
    ppc_cpu_init(host_bridge, PPC_VER::MPC750, false, 16705000,
                 233870000);
    TimerManager::get_instance()->cancel_all_timers();

    cy_check(icnt_factor == 2,
             "a 233.87 MHz G3 selects the nearest 250 MIPS timing rate");

    test_deadline();
    test_batching();
    test_realtime_keeps_virtual_time();

    TimerManager::get_instance()->cancel_all_timers();

    cout << "Tested " << dec << cy_tested << " cycle accounting behaviours. Failed: "
         << cy_failed << "." << endl;

    return cy_failed;
}
