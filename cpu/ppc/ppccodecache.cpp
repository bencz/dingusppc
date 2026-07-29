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

/** @file Bookkeeping for translated code blocks. */

#include <loguru.hpp>
#include "ppccodecache.h"
#include "ppcmmu.h"

#include <unordered_map>
#include <utility>
#include <vector>

unsigned ppc_code_cache_count = 0;

namespace {

typedef struct CodeBlock {
    uint32_t        start;  // physical address of the first instruction
    uint32_t        end;    // physical address just past the last one
    CodeBlockHandle handle;
} CodeBlock;

/* Blocks never cross a page, so grouping them by page turns invalidation of
   a cache line into a lookup plus a scan over a handful of entries. */
std::unordered_map<uint32_t, std::vector<CodeBlock>> blocks_by_page;

std::function<void(CodeBlockHandle)> release_cb;

/** Clamps to the end of the address space so a wrapping size cannot turn
    into an empty range */
inline uint32_t range_last(uint32_t phys_addr, uint32_t size) {
    uint64_t last = uint64_t(phys_addr) + size - 1;
    return last > 0xFFFFFFFFULL ? 0xFFFFFFFFUL : uint32_t(last);
}

} // namespace

void ppc_code_cache_init() {
    ppc_code_cache_invalidate_all();
    release_cb = nullptr;
}

void ppc_code_cache_reset() {
    ppc_code_cache_invalidate_all();
}

void ppc_code_cache_set_release_cb(std::function<void(CodeBlockHandle)> cb) {
    release_cb = std::move(cb);
}

void ppc_code_cache_add(uint32_t phys_addr, uint32_t size, CodeBlockHandle handle) {
    if (!size) {
        return;
    }

    uint32_t page = phys_addr & PPC_PAGE_MASK;
    if ((range_last(phys_addr, size) & PPC_PAGE_MASK) != page) {
        ABORT_F("Code block at 0x%08X spans more than one page", phys_addr);
    }

    auto& page_blocks = blocks_by_page[page];
    bool  was_empty   = page_blocks.empty();

    page_blocks.push_back({phys_addr, phys_addr + size, handle});
    ppc_code_cache_count++;

    // stores have to be noticed from now on, but only the transition matters
    if (was_empty) {
        mmu_mark_code_page(page);
    }
}

bool ppc_code_cache_page_has_blocks(uint32_t phys_addr) {
    if (ppc_code_cache_is_empty()) {
        return false;
    }
    return blocks_by_page.find(phys_addr & PPC_PAGE_MASK) != blocks_by_page.end();
}

unsigned ppc_code_cache_invalidate(uint32_t phys_addr, uint32_t size) {
    if (ppc_code_cache_is_empty() || !size) {
        return 0;
    }

    const uint32_t last      = range_last(phys_addr, size);
    const uint32_t last_page = last & PPC_PAGE_MASK;
    unsigned       dropped   = 0;

    for (uint32_t page = phys_addr & PPC_PAGE_MASK;; page += PPC_PAGE_SIZE) {
        auto it = blocks_by_page.find(page);
        if (it != blocks_by_page.end()) {
            auto& page_blocks = it->second;
            for (size_t i = 0; i < page_blocks.size();) {
                const CodeBlock& blk = page_blocks[i];
                if (blk.start <= last && blk.end > phys_addr) {
                    if (release_cb) {
                        release_cb(blk.handle);
                    }
                    page_blocks[i] = page_blocks.back();
                    page_blocks.pop_back();
                    dropped++;
                } else {
                    i++;
                }
            }
            if (page_blocks.empty()) {
                blocks_by_page.erase(it);
            }
        }
        if (page == last_page) {
            break;
        }
    }

    ppc_code_cache_count -= dropped;
    return dropped;
}

unsigned ppc_code_cache_invalidate_all() {
    unsigned dropped = ppc_code_cache_count;

    if (release_cb) {
        for (auto& page : blocks_by_page) {
            for (auto& blk : page.second) {
                release_cb(blk.handle);
            }
        }
    }

    blocks_by_page.clear();
    ppc_code_cache_count = 0;
    return dropped;
}
