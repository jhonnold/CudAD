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

#ifndef CUDATEST1_SRC_OPERATIONS_MAT_MAT_PRODUCT_MAT_MAT_PRODUCT_CUBLAS_H_
#define CUDATEST1_SRC_OPERATIONS_MAT_MAT_PRODUCT_MAT_MAT_PRODUCT_CUBLAS_H_

#include "../../data/DenseMatrix.h"
#include "../../misc/config.h"
// clang-format off
inline void mm_cublas(
    const DenseMatrix &A,
    const DenseMatrix &B,
          DenseMatrix &C,
    const float alpha = 1,
    const float beta = 0,
    bool transpose_A=false,
    bool transpose_B=false) {
    // clang-format on

    const int m       = C.m;
    const int n       = C.n;
    const int k       = transpose_A ? A.m : A.n;

    int       lda     = A.leading_dimension;
    int       ldb     = B.leading_dimension;
    int       ldc     = C.leading_dimension;

    auto      trans_a = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto      trans_b = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    // clang-format off
    cublasSgemm(CUBLAS_HANDLE, trans_a, trans_b,
                m, n, k, &alpha,
                A.gpu_values, lda,
                B.gpu_values, ldb, &beta,
                C.gpu_values, ldc);
    // clang-format on

    CUDA_ASSERT(cudaPeekAtLastError());

    //    cudaDeviceSynchronize();
}

#endif    // CUDATEST1_SRC_OPERATIONS_MAT_MAT_PRODUCT_MAT_MAT_PRODUCT_CUBLAS_H_
