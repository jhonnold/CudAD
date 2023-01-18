/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

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

#ifndef POSITION_ZOBRIST_H
#define POSITION_ZOBRIST_H

#include "defs.h"
#include "position.h"

#include <chrono>

class PRNG {
    uint64_t s;

    uint64_t rand64() {
        s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
        return s * 2685821657736338717LL;
    }

    public:
    PRNG() { s = 1070372; }
    
    PRNG(uint64_t seed) { s = seed; }

    template<typename T>
    T rand() {
        return T(rand64());
    }
};

class Zobrist {
    PRNG prng;

    Key  psq[16][64];
    Key  ep[64];
    Key  castling[16];
    Key  side;

    public:
    Zobrist(PRNG prng) {
        for (Piece pc : {WHITE_PAWN,
                         WHITE_KNIGHT,
                         WHITE_BISHOP,
                         WHITE_ROOK,
                         WHITE_QUEEN,
                         WHITE_KING,
                         BLACK_PAWN,
                         BLACK_KNIGHT,
                         BLACK_BISHOP,
                         BLACK_ROOK,
                         BLACK_QUEEN,
                         BLACK_KING})
            for (Square sq = A1; sq <= H8; sq++)
                psq[pc][sq] = prng.rand<Key>();

        for (Square sq = A1; sq <= H8; sq++)
            ep[sq] = prng.rand<Key>();

        for (int cr = 0; cr < 16; cr++)
            castling[cr] = prng.rand<Key>();

        side = prng.rand<Key>();
    }

    Key get_key(Position& pos) {
        Key    key = pos.m_meta.getActivePlayer() == WHITE ? 0 : side;

        size_t idx = 0;
        BB     occ = pos.m_occupancy;
        while (occ) {
            Square sq = bitscanForward(occ);
            Piece  pc = pos.m_pieces.getPiece(idx);

            key ^= psq[pc][sq];

            occ = lsbReset(occ);
            idx++;
        }

        Square epsq = pos.m_meta.getEnPassantSquare();
        if (epsq != 0) key ^= ep[epsq];

        uint8_t cr = pos.m_meta.m_castling_and_active_player & 0x0F; // bottom 4 bits are castling rights
        return key ^ castling[cr];
    }
};

#endif