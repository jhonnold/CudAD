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

#include "sq_relu_bp.h"

// clang-format off
__global__ void sq_relu_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int size){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    if (B[idx] > 0) {
        A_grd[idx] = 2 * A[idx] * B_grd[idx];
    } else {
        A_grd[idx] = 0;
    }
}
