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
#include "ppcemu.h"
#include "ppcmmu.h"

#include <algorithm>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

std::atomic<unsigned> ppc_code_cache_count{0};

namespace {

typedef struct CodeBlock {
    uint32_t        start;  // physical address of the first instruction
    uint32_t        end;    // physical address just past the last one
    CodeBlockHandle handle;
} CodeBlock;

/* Blocks never cross a page, so grouping them by page turns invalidation of
   a cache line into a lookup plus a scan over a handful of entries. */
std::unordered_map<uint32_t, std::vector<CodeBlock>> blocks_by_page;

/* Physical pages whose data mappings have been told to route stores through
   the slow path, where an overwrite of translated code is noticed.

   A page enters the set with its first block and leaves it when a store
   finds it empty of blocks, which is the only moment writability can be
   handed back without a window where new code sits unguarded. Marking is an
   epoch bump in the MMU, so neither direction walks anything. */
std::unordered_set<uint32_t> protected_pages;

std::function<void(CodeBlockHandle)> release_cb;

struct InvalidationRange {
    uint32_t start;
    uint32_t size;
};

/* Cubeb drives sound DBDMA from its callback thread. That thread may map and
   update DBDMA descriptors while the CPU is compiling or following a direct
   chain. It must never walk blocks_by_page or invoke the JIT release callback:
   neither the registry nor the chain-link lists are concurrent containers. */
std::thread::id                 code_cache_owner_thread;
std::mutex                      dma_invalidations_mutex;
std::vector<InvalidationRange>  dma_invalidations;

/** Clamps to the end of the address space so a wrapping size cannot turn
    into an empty range */
inline uint32_t range_last(uint32_t phys_addr, uint32_t size) {
    uint64_t last = uint64_t(phys_addr) + size - 1;
    return last > 0xFFFFFFFFULL ? 0xFFFFFFFFUL : uint32_t(last);
}

void clear_dma_invalidations() {
    std::lock_guard<std::mutex> lock(dma_invalidations_mutex);
    dma_invalidations.clear();
}

} // namespace

void ppc_code_cache_init() {
    ppc_code_cache_invalidate_all();
    clear_dma_invalidations();
    code_cache_owner_thread = std::this_thread::get_id();

    // the TLBs are being rebuilt around this call anyway, so nothing is
    // left protected that should not be
    protected_pages.clear();
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

    blocks_by_page[page].push_back({phys_addr, phys_addr + size, handle});
    ppc_code_cache_count.fetch_add(1, std::memory_order_relaxed);

    // stores have to be noticed from now on. Only the first block ever put on
    // the page pays for saying so, see protected_pages
    if (protected_pages.insert(page).second) {
        mmu_mark_code_page(page);
    }
}

bool ppc_code_cache_page_is_protected(uint32_t phys_addr) {
    if (protected_pages.empty()) {
        return false;
    }
    return protected_pages.find(phys_addr & PPC_PAGE_MASK) != protected_pages.end();
}

void ppc_code_cache_remove(uint32_t phys_addr, CodeBlockHandle handle) {
    auto it = blocks_by_page.find(phys_addr & PPC_PAGE_MASK);
    if (it == blocks_by_page.end()) {
        return;
    }
    for (CodeBlock& blk : it->second) {
        if (blk.handle == handle) {
            blk.handle = nullptr;
            ppc_code_cache_count.fetch_sub(1, std::memory_order_relaxed);
            return;
        }
    }
}

bool ppc_code_cache_store_to_page(uint32_t phys_addr, uint32_t size) {
    ppc_code_cache_invalidate(phys_addr, size);

    const uint32_t page = phys_addr & PPC_PAGE_MASK;
    if (blocks_by_page.find(page) == blocks_by_page.end()) {
        protected_pages.erase(page);
        return false;
    }
    return true;
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
                if (!blk.handle) {
                    // a tombstone left by ppc_code_cache_remove; the count
                    // was already adjusted when it died
                    page_blocks[i] = page_blocks.back();
                    page_blocks.pop_back();
                } else if (blk.start <= last && blk.end > phys_addr) {
                    // Retire this registration before notifying its owner.
                    // A block spanning two translated regions uses the same
                    // handle twice, and its callback tombstones both entries.
                    // Leaving the current entry live until after the callback
                    // made ppc_code_cache_remove decrement it here and then
                    // made this walk decrement it a second time, wrapping the
                    // unsigned global count and leaving the cache permanently
                    // non-empty after a cross-region invalidation.
                    const CodeBlockHandle handle = blk.handle;
                    page_blocks[i].handle = nullptr;
                    ppc_code_cache_count.fetch_sub(1, std::memory_order_relaxed);
                    if (release_cb) {
                        release_cb(handle);
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

    return dropped;
}

void ppc_code_cache_invalidate_dma(uint32_t phys_addr, uint32_t size) {
    if (!size || ppc_code_cache_is_empty()) {
        return;
    }

    if (std::this_thread::get_id() == code_cache_owner_thread) {
        ppc_code_cache_invalidate(phys_addr, size);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(dma_invalidations_mutex);

        // DMA descriptors and buffers are normally consumed in order. Merge
        // with the last range when possible to keep a busy device from
        // growing the queue between two CPU event checks.
        if (!dma_invalidations.empty()) {
            InvalidationRange& prev = dma_invalidations.back();
            const uint64_t address_space_end = uint64_t(UINT32_MAX) + 1;
            const uint64_t prev_end = std::min(
                uint64_t(prev.start) + prev.size, address_space_end);
            const uint64_t new_end = std::min(
                uint64_t(phys_addr) + size, address_space_end);
            if (uint64_t(phys_addr) <= prev_end && uint64_t(prev.start) <= new_end) {
                const uint64_t merged_start = std::min(prev.start, phys_addr);
                const uint64_t merged_size  = std::max(prev_end, new_end) - merged_start;
                if (merged_size <= UINT32_MAX) {
                    prev.start = uint32_t(merged_start);
                    prev.size  = uint32_t(merged_size);
                } else {
                    dma_invalidations.push_back({phys_addr, size});
                }
            } else {
                dma_invalidations.push_back({phys_addr, size});
            }
        } else {
            dma_invalidations.push_back({phys_addr, size});
        }
    }

    // The generated-code exit checks this byte before following another
    // chain, so the CPU reaches ppc_process_events and drains the queue.
    exec_timer = true;
}

unsigned ppc_code_cache_drain_dma_invalidations() {
    std::vector<InvalidationRange> pending;
    {
        std::lock_guard<std::mutex> lock(dma_invalidations_mutex);
        pending.swap(dma_invalidations);
    }

    unsigned dropped = 0;
    for (const InvalidationRange& range : pending) {
        dropped += ppc_code_cache_invalidate(range.start, range.size);
    }
    return dropped;
}

unsigned ppc_code_cache_invalidate_all() {
    unsigned dropped = ppc_code_cache_count.load(std::memory_order_relaxed);

    if (release_cb) {
        for (auto& page : blocks_by_page) {
            for (auto& blk : page.second) {
                if (blk.handle) {
                    release_cb(blk.handle);
                }
            }
        }
    }

    blocks_by_page.clear();
    ppc_code_cache_count.store(0, std::memory_order_relaxed);
    return dropped;
}
