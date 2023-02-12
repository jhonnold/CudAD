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

#ifndef DIFFERENTIATION_DUPLICATEDENSELAYER_H
#define DIFFERENTIATION_DUPLICATEDENSELAYER_H

#include "../operations/operations.h"
#include "Layer.h"

template<int I, int O, typename F>
class DuplicateDenseLayer : public LayerInterface {
    public:
    Tape weights {O, I};
    Tape bias {O, 1};
    F    f {};

    // regularization
    float lasso_regularization = 0;

    explicit DuplicateDenseLayer(int expected_active_inputs = I) {
        weights.values.randomiseKaiming(expected_active_inputs);

        weights.values.gpu_upload();
        bias.values.gpu_upload();
    }

    void apply(std::vector<Tape*> inputs, Tape& out) override {
        //        mat_mat_product<MODE_GPU>(weights, in, out);
        //        vm_add<MODE_GPU>(out, bias, out);
        //        f.apply(out, out, MODE_GPU);
        ASSERT(false);
    }
    void apply(std::vector<SparseInput*> inputs, Tape& out) override {
        uint32_t B = out.values.n;
        // create submatrices for the output
        DenseMatrix mat_res_1 {out.values, 0, 0, O, B};
        DenseMatrix mat_res_2 {out.values, O, 0, O, B};
        sparse_affine<DEVICE>(weights.values, *inputs[0], bias.values, mat_res_1);
        sparse_affine<DEVICE>(weights.values, *inputs[1], bias.values, mat_res_2);
        f.apply(out.values, out.values, DEVICE);
    }

    void backprop(std::vector<Tape*> inputs, Tape& out) override { ASSERT(false); }

    void backprop(std::vector<SparseInput*> inputs, Tape& out) override {
        uint32_t B = out.values.n;

        f.backprop(out.values, out.gradients, out.values, out.gradients, DEVICE);

        // create submatrices for the output
        // clang-format off
        DenseMatrix mat_grd_1 {out.gradients, 0, 0, O, B};
        DenseMatrix mat_grd_2 {out.gradients, O, 0, O, B};
        DenseMatrix mat_res_1 {out.values   , 0, 0, O, B};
        DenseMatrix mat_res_2 {out.values   , O, 0, O, B};
        // clang-format on

        sparse_affine_bp<DEVICE>(weights.gradients,
                                 *inputs[0],
                                 bias.gradients,
                                 mat_res_1,
                                 mat_grd_1,
                                 lasso_regularization);
        sparse_affine_bp<DEVICE>(weights.gradients,
                                 *inputs[1],
                                 bias.gradients,
                                 mat_res_2,
                                 mat_grd_2,
                                 lasso_regularization);
    }

    uint32_t           getOutputSize() override { return O * 2; }
    uint32_t           getInputSize() override { return I * 2; }
    std::vector<Tape*> getTunableParameters() override {
        return std::vector<Tape*> {&weights, &bias};
    }
    Activation* getActivationFunction() override { return &f; }
};

#endif    // DIFFERENTIATION_DUPLICATEDENSELAYER_H
