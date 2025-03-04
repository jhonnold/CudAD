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

#ifndef CUDAD_SRC_NETWORK_NETWORK_H_
#define CUDAD_SRC_NETWORK_NETWORK_H_

#include "../layer/Layer.h"
#include "../loss/Loss.h"
#include "../operations/mse/mse.h"

#include <utility>
class Network {

    private:
    std::vector<LayerInterface*> layers {};
    std::vector<Tape>            output_tapes {};
    Loss*                        loss_function {};

    public:
    explicit Network(std::vector<LayerInterface*> layers) : layers(std::move(layers)) {}

    void                         createOutputTapes(int batch_size);

    void                         batch(const std::vector<SparseInput*>& inputs,
                                       const DenseMatrix&               target,
                                       const SArray<bool>&              target_mask);

    void                         batch(const std::vector<Tape*>& inputs,
                                       const DenseMatrix&        target,
                                       const SArray<bool>&       target_mask);

    void                         feed(const std::vector<SparseInput*>& inputs);

    void                         feed(const std::vector<Tape*>& inputs);

    Loss*                        getLossFunction() const;
    void                         setLossFunction(Loss* loss_function);

    void                         loadWeights(const std::string& file);

    void                         saveWeights(const std::string& file);

    Tape&                        getOutput();
    std::vector<LayerInterface*> getLayers();

    Tape&                        getOutput(int layer_id);
};

#endif    // CUDAD_SRC_NETWORK_NETWORK_H_
