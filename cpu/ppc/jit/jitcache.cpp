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

/** @file Lookup side of the translated code. */

#include "jitcache.h"

#include <unordered_map>

namespace dppc_jit {

namespace {

std::unordered_map<uint64_t, JitBlock*> blocks;

/** Direct mapped accelerator in front of the map above.

    Measured, not assumed: with blocks averaging 4.3 guest instructions the
    dispatch between them dominates, and hashing a 64 bit key and chasing a
    bucket is most of it. This turns the common case into an index, two loads
    and two compares.

    It caches, it does not own. The map stays the authority. A matching entry
    with a null block is a cached map miss, which keeps every cold interpreter
    entry from repeating the same hash lookup. PPC_JIT_MODE_MASK makes all
    ones an impossible mode, so it marks a slot whose key is not valid.

    Keyed by physical address like the map, which is what keeps it free of the
    invalidation hazards a virtual key would bring: a virtual address can
    start meaning different memory without any block being dropped */
constexpr uint32_t QUICK_INVALID_MODE = ~uint32_t(0);

struct QuickEntry {
    uint32_t  phys_addr = 0;
    uint32_t  mode      = QUICK_INVALID_MODE;
    JitBlock* blk       = nullptr;
};

constexpr size_t QUICK_BITS = 13;
constexpr size_t QUICK_SIZE = size_t(1) << QUICK_BITS;
constexpr size_t QUICK_MASK = QUICK_SIZE - 1;

QuickEntry quick[QUICK_SIZE];

inline size_t quick_index(uint32_t phys_addr) {
    // blocks start on an instruction boundary, so the low two bits carry
    // nothing and would only halve the useful size of the table. Growing the
    // table to 17 bits was measured and bought nothing: the cost is how often
    // lookup runs, not how often it collides
    return (phys_addr >> 2) & QUICK_MASK;
}

void quick_clear() {
    for (size_t i = 0; i < QUICK_SIZE; i++) {
        quick[i].mode = QUICK_INVALID_MODE;
        quick[i].blk  = nullptr;
    }
}

} // namespace

JitBlock* cache_lookup(uint32_t phys_addr, uint32_t mode) {
    QuickEntry& q = quick[quick_index(phys_addr)];
    if (q.phys_addr == phys_addr && q.mode == mode) [[likely]] {
        return q.blk;
    }

    auto it = blocks.find(block_key(phys_addr, mode));
    if (it == blocks.end()) {
        q.phys_addr = phys_addr;
        q.mode      = mode;
        q.blk       = nullptr;
        return nullptr;
    }

    q.phys_addr = phys_addr;
    q.mode      = mode;
    q.blk       = it->second;
    return it->second;
}

void cache_insert(JitBlock* blk) {
    blocks[block_key(blk->phys_addr, blk->mode)] = blk;

    QuickEntry& q = quick[quick_index(blk->phys_addr)];
    q.phys_addr = blk->phys_addr;
    q.mode      = blk->mode;
    q.blk       = blk;
}

void cache_forget(JitBlock* blk) {
    auto it = blocks.find(block_key(blk->phys_addr, blk->mode));
    // only erase the entry if it is still this block, a newer translation of
    // the same address may already have taken the slot
    if (it != blocks.end() && it->second == blk) {
        blocks.erase(it);
    }

    // insertion always lands a block on the slot its physical address picks,
    // so that is the only slot that can be holding it. Clearing the whole
    // table instead was measured at a fifth of the time of a real boot: a
    // page being written drops every block on it, and each of those was
    // walking all 8192 entries
    QuickEntry& q = quick[quick_index(blk->phys_addr)];
    if (q.blk == blk) {
        // The authoritative entry was just erased, so this key is now a
        // known miss. A later translation overwrites it in cache_insert.
        q.blk = nullptr;
    }
}

void cache_clear() {
    blocks.clear();
    quick_clear();
}

size_t cache_size() {
    return blocks.size();
}

std::vector<JitBlock*> cache_blocks() {
    std::vector<JitBlock*> result;
    result.reserve(blocks.size());
    for (const auto& entry : blocks) {
        result.push_back(entry.second);
    }
    return result;
}

} // namespace dppc_jit
