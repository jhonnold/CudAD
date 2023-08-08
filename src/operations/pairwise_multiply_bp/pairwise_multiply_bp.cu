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

#include "pairwise_multiply_bp.h"

// clang-format off
__global__ void pairwise_multiply_bp_kernel(
    const float* __restrict__ input,
          float* __restrict__ input_grd,
    const float* __restrict__ output_grd,
    unsigned int outsize,
    unsigned int neurons){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= outsize)
        return;

    int halfsize = neurons / 2;

    int idx1 = idx + halfsize * (idx / halfsize);
    int idx2 = idx1 + halfsize;

    input_grd[idx1] = output_grd[idx] * input[idx2];
    input_grd[idx2] = output_grd[idx] * input[idx1];
}