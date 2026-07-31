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

/** @file Storage side of the JIT intermediate representation. */

#include "jitir.h"

namespace dppc_jit {

uint32_t jit_max_block_insns = JIT_MAX_BLOCK_INSNS;
uint32_t jit_decode_groups   = JIT_DECODE_ALL;
bool     jit_sync_every_call = false;
bool     jit_superblocks     = true;
bool     jit_cross_follow    = true;
bool     jit_blr_follow      = true;
bool     jit_cr_fuse         = true;
bool     jit_pool_recycle    = false;

const char* block_end_name(BlockEnd reason) {
    switch (reason) {
    case BlockEnd::Branch:         return "branch";
    case BlockEnd::ContextSync:    return "context sync";
    case BlockEnd::PageEnd:        return "page end";
    case BlockEnd::Untranslatable: return "untranslatable";
    case BlockEnd::SizeLimit:      return "size limit";
    }
    return "?";
}

void IRBlock::reset(uint32_t virt, uint32_t phys, uint32_t translation_mode) {
    this->virt_addr  = virt;
    this->phys_addr  = phys;
    this->mode       = translation_mode;
    this->byte_size   = 0;
    this->end_off     = 0;
    this->second_phys = 0;
    this->second_off  = 0;
    this->second_size = 0;
    this->insn_count  = 0;
    this->end_reason  = BlockEnd::SizeLimit;
    this->end_word    = 0;

    // keeps the capacity, translation runs often enough that the reuse matters
    this->insns.clear();
}

IRValue IRBlock::append(const IRInsn& insn) {
    this->insns.push_back(insn);

    // SSA: the defining instruction is the value, so its index names it
    return IRValue(this->insns.size() - 1);
}

} // namespace dppc_jit
