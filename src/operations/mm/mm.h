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

#ifndef CUDAD_SRC_OPERATIONS_MM_MM_H_
#define CUDAD_SRC_OPERATIONS_MM_MM_H_

#include "mm_cublas.h"

template<Mode mode>
inline void mm(DenseMatrix& mat1, DenseMatrix& mat2, DenseMatrix& res) {
    ASSERT(mat1.n == mat2.m);
    ASSERT(mat1.m == res.m);
    ASSERT(mat2.n == res.n);

    if (mode == DEVICE) {
        mm_cublas(mat1, mat2, res);
    } else {
        ASSERT(false);
    }
}

#endif    // CUDAD_SRC_OPERATIONS_MM_MM_H_
