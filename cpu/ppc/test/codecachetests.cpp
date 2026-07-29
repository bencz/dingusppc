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

/** @file Tests for the translated block registry and for icbi.

    These run last because the icbi part needs a memory controller and calls
    ppc_cpu_init, which resets the processor state the other tests set up.
 */

#include "../ppccodecache.h"
#include "../ppcemu.h"
#include "../ppcmmu.h"
#include "devices/memctrl/mpc106.h"

#include <iostream>
#include <vector>

using namespace std;

// icbi 0,r3
constexpr uint32_t OPCODE_ICBI_R3 = 0x7C001FAC;

static int cc_tested;
static int cc_failed;

static void cc_check(bool passed, const char* what) {
    cc_tested++;
    if (!passed) {
        cout << "  Failed: " << what << endl;
        cc_failed++;
    }
}

static void test_registry() {
    ppc_code_cache_init();
    cc_check(ppc_code_cache_is_empty(), "cache starts out empty");

    ppc_code_cache_add(0x1000, 0x40, (CodeBlockHandle)1);
    ppc_code_cache_add(0x1040, 0x40, (CodeBlockHandle)2);
    ppc_code_cache_add(0x2000, 0x40, (CodeBlockHandle)3);
    cc_check(ppc_code_cache_num_blocks() == 3, "three blocks got registered");
    cc_check(!ppc_code_cache_is_empty(), "cache reports itself non empty");

    cc_check(ppc_code_cache_invalidate(0x1020, PPC_ICACHE_LINE_SIZE) == 1,
             "a line inside a block drops that block alone");
    cc_check(ppc_code_cache_num_blocks() == 2, "the other two blocks survive");

    cc_check(ppc_code_cache_invalidate(0x3000, PPC_ICACHE_LINE_SIZE) == 0,
             "a line on an untracked page drops nothing");
    cc_check(ppc_code_cache_invalidate(0x1FE0, PPC_ICACHE_LINE_SIZE) == 0,
             "a line past the end of a block drops nothing");

    ppc_code_cache_add(0x1000, 0x40, (CodeBlockHandle)4);
    cc_check(ppc_code_cache_invalidate(0x1000, 0x1100) == 3,
             "a range across two pages reaches blocks on both");
    cc_check(ppc_code_cache_is_empty(), "the cache is empty again");
}

static void test_release_callback() {
    vector<CodeBlockHandle> released;

    ppc_code_cache_set_release_cb([&](CodeBlockHandle handle) {
        released.push_back(handle);
    });

    ppc_code_cache_add(0x5000, PPC_ICACHE_LINE_SIZE, (CodeBlockHandle)7);
    ppc_code_cache_add(0x6000, PPC_ICACHE_LINE_SIZE, (CodeBlockHandle)8);

    ppc_code_cache_invalidate(0x5000, PPC_ICACHE_LINE_SIZE);
    cc_check(released.size() == 1 && released[0] == (CodeBlockHandle)7,
             "the callback fires for the dropped block only");

    cc_check(ppc_code_cache_invalidate_all() == 1, "a full flush drops what is left");
    cc_check(released.size() == 2, "the callback fires on a full flush too");

    ppc_code_cache_set_release_cb(nullptr);
}

static void test_icbi() {
    MPC106* host_bridge = new MPC106;

    if (!host_bridge->add_ram_region(0, 0x10000)) {
        cc_check(false, "could not create a RAM region for the icbi test");
        delete host_bridge;
        return;
    }

    ppc_cpu_init(host_bridge, PPC_VER::MPC750, false, 16705000);

    // MSR[DR] is clear coming out of reset, so effective addresses are physical
    ppc_code_cache_add(0x2000, PPC_ICACHE_LINE_SIZE, (CodeBlockHandle)9);
    ppc_code_cache_add(0x4000, PPC_ICACHE_LINE_SIZE, (CodeBlockHandle)10);

    ppc_state.gpr[3] = 0x2010; // partway into the first block
    ppc_main_opcode(ppc_opcode_grabber, OPCODE_ICBI_R3);
    cc_check(ppc_code_cache_num_blocks() == 1, "icbi drops the block under its line");

    ppc_state.gpr[3] = 0x4000;
    ppc_main_opcode(ppc_opcode_grabber, OPCODE_ICBI_R3);
    cc_check(ppc_code_cache_is_empty(), "icbi drops the second block as well");

    // an address with nothing registered has to be harmless
    ppc_state.gpr[3] = 0x8000;
    ppc_main_opcode(ppc_opcode_grabber, OPCODE_ICBI_R3);
    cc_check(ppc_code_cache_is_empty(), "icbi on an untracked line stays quiet");
}

/** Runs after test_icbi, reusing the machine it brought up */
static void test_store_invalidation() {
    ppc_code_cache_reset();

    ppc_code_cache_add(0x2000, PPC_ICACHE_LINE_SIZE, (CodeBlockHandle)11);
    ppc_code_cache_add(0x2800, PPC_ICACHE_LINE_SIZE, (CodeBlockHandle)12);
    ppc_code_cache_add(0x3000, PPC_ICACHE_LINE_SIZE, (CodeBlockHandle)13);
    cc_check(ppc_code_cache_num_blocks() == 3, "three blocks spread over two pages");

    // a store anywhere on a code page takes down every block on that page
    mmu_write_vmem<uint32_t>(NO_OPCODE, 0x2FF0, 0xDEADBEEF);
    cc_check(ppc_code_cache_num_blocks() == 1, "a store drops both blocks on its page");
    cc_check(ppc_code_cache_page_has_blocks(0x3000), "the untouched page keeps its block");
    cc_check(mmu_read_vmem<uint32_t>(NO_OPCODE, 0x2FF0) == 0xDEADBEEF,
             "the store that triggered the drop still went through");

    // the page is ordinary memory again now
    mmu_write_vmem<uint32_t>(NO_OPCODE, 0x2000, 0x12345678);
    cc_check(ppc_code_cache_num_blocks() == 1, "a later store on that page drops nothing");
    cc_check(mmu_read_vmem<uint32_t>(NO_OPCODE, 0x2000) == 0x12345678,
             "the later store landed as well");

    // a store nowhere near a code page changes nothing
    mmu_write_vmem<uint32_t>(NO_OPCODE, 0x5000, 0xA5A5A5A5);
    cc_check(ppc_code_cache_num_blocks() == 1, "a store off any code page drops nothing");

    mmu_write_vmem<uint32_t>(NO_OPCODE, 0x3010, 0);
    cc_check(ppc_code_cache_is_empty(), "storing on the last code page empties the cache");
}

int test_code_cache() {
    cc_tested = 0;
    cc_failed = 0;

    test_registry();
    test_release_callback();
    test_icbi();
    test_store_invalidation();

    ppc_code_cache_init();

    cout << "Tested " << dec << cc_tested << " code cache behaviours. Failed: "
         << cc_failed << "." << endl;

    return cc_failed;
}
