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

#include "../operations/mle/mle.h"
#include "Network.h"

void Network::loadWeights(const std::string& file) {
    FILE* f = fopen(file.c_str(), "rb");

    // figure out how many entries we will store
    uint64_t count = 0;
    for (LayerInterface* l : layers) {
        for (Tape* t : l->getTunableParameters()) {
            count += t->values.size;
        }
    }

    uint64_t fileCount = 0;
    fread(&fileCount, sizeof(uint64_t), 1, f);
    ASSERT(count == fileCount);

    for (LayerInterface* l : layers) {
        for (Tape* t : l->getTunableParameters()) {
            fread(t->values.cpu_values, sizeof(float), t->values.size, f);
            t->values.gpu_upload();
        }
    }
    fclose(f);
}
void Network::saveWeights(const std::string& file) {
    FILE* f = fopen(file.c_str(), "wb");

    // figure out how many entries we will store
    uint64_t count = 0;
    for (LayerInterface* l : layers) {
        for (Tape* t : l->getTunableParameters()) {
            count += t->values.size;
        }
    }

    fwrite(&count, sizeof(uint64_t), 1, f);
    for (LayerInterface* l : layers) {
        for (Tape* t : l->getTunableParameters()) {
            t->values.gpu_download();
            fwrite(t->values.cpu_values, sizeof(float), t->values.size, f);
        }
    }
    fclose(f);
}
void Network::batch(const std::vector<SparseInput*>& inputs,
                    const DenseMatrix&               target,
                    const SArray<bool>&              target_mask) {
    feed(inputs);

    loss_function->apply(output_tapes[output_tapes.size() - 1].values,
                         output_tapes[output_tapes.size() - 1].gradients,
                         target,
                         target_mask,
                         DEVICE);

    for (int i = layers.size() - 1; i >= 1; i--) {
        layers[i]->backprop({&output_tapes[i - 1]}, output_tapes[i]);
    }
    layers[0]->backprop(inputs, output_tapes[0]);
}

void Network::batch(const std::vector<Tape*>& inputs,
                    const DenseMatrix&        target,
                    const SArray<bool>&       target_mask) {
    feed(inputs);

    loss_function->apply(output_tapes[output_tapes.size() - 1].values,
                         output_tapes[output_tapes.size() - 1].gradients,
                         target,
                         target_mask,
                         DEVICE);

    for (int i = layers.size() - 1; i >= 1; i--) {
        layers[i]->backprop({&output_tapes[i - 1]}, output_tapes[i]);
    }
    layers[0]->backprop(inputs, output_tapes[0]);
}

void Network::feed(const std::vector<SparseInput*>& inputs) {
    createOutputTapes(inputs[0]->n);
    layers[0]->apply(inputs, output_tapes[0]);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->apply({&output_tapes[i - 1]}, output_tapes[i]);
    }
}

void Network::feed(const std::vector<Tape*>& inputs) {
    createOutputTapes(inputs[0]->values.n);
    layers[0]->apply(inputs, output_tapes[0]);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->apply({&output_tapes[i - 1]}, output_tapes[i]);
    }
}

void Network::createOutputTapes(int batch_size) {
    // if there are tapes already with the correct size, dont create new tapes
    if (!output_tapes.empty() && output_tapes[0].values.n == batch_size)
        return;
    // clear the tapes
    output_tapes.clear();
    // create a mew tape
    for (int i = 0; i < layers.size(); i++) {
        output_tapes.emplace_back((uint32_t) layers[i]->getOutputSize(),    // output of layer
                                  (uint32_t) batch_size);                   // batch size
    }
}
Tape&                        Network::getOutput() { return output_tapes[output_tapes.size() - 1]; }
std::vector<LayerInterface*> Network::getLayers() { return layers; }
Tape&                        Network::getOutput(int layer_id) { return output_tapes[layer_id]; }
Loss*                        Network::getLossFunction() const { return loss_function; }
void Network::setLossFunction(Loss* loss_function) { Network::loss_function = loss_function; }
