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

/** @file Lookup side of the translated code.

    The other half is ppccodecache.h, which owns invalidation and is keyed by
    physical address alone, because guest memory being overwritten kills every
    translation of it no matter which mode produced them. This is the side the
    generator needs: given a physical address and the current mode, find the
    block that matches both.

    Splitting them this way is what lets ppccodecache stay ignorant of the
    generator and keep working, empty, in a build without one.
 */

#ifndef PPC_JIT_CACHE_H
#define PPC_JIT_CACHE_H

#include "jitblock.h"

#include <cstddef>

namespace dppc_jit {

JitBlock* cache_lookup(uint32_t phys_addr, uint32_t mode);

void cache_insert(JitBlock* blk);

/** Unlinks a block without releasing it. Called from the invalidation path,
    where the block may well be the one running right now */
void cache_forget(JitBlock* blk);

void cache_clear();

size_t cache_size();

} // namespace dppc_jit

#endif // PPC_JIT_CACHE_H
