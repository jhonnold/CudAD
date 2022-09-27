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

#include "archs/Berserk.h"
#include "misc/config.h"
#include "trainer.h"
#include "data/DenseMatrix.h"
#include "data/SArray.h"
#include "data/SparseInput.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "loss/Loss.h"
#include "misc/csv.h"
#include "misc/timer.h"
#include "network/Network.h"
#include "optimizer/Optimiser.h"
#include "quantitize.h"
#include "dataset/writer.h"

#include <iostream>
#include <vector>

constexpr int BatchSize = 16384;
constexpr int BatchesPerFile = 6103;
constexpr int TotalFiles = 74;

using namespace std;

int main() {
    init();

    const string data_path = "E:/berserk/training-data/master/";
    const string output    = "./resources/runs/exp100/";
    vector<string> files {};
    for (int i = 0; i < 20; i++)
        files.push_back(data_path + "n5k." + to_string(i) + ".bin");
    
    BatchLoader batch_loader {files, BatchSize};
    batch_loader.start();


    tuple<SparseInput, SparseInput> inputs {SparseInput {Berserk::Inputs, BatchSize, 32},
                                            SparseInput {Berserk::Inputs, BatchSize, 32}};
    DenseMatrix                     target {Berserk::Outputs, BatchSize};
    SArray<bool>                    target_mask {Berserk::Outputs * BatchSize};

    target_mask.malloc_cpu();
    target_mask.malloc_gpu();

    vector<LayerInterface*> layers = Berserk::get_layers();
    Network* network = new Network(layers);
    network->setLossFunction(Berserk::get_loss_function());
    network->loadWeights("./resources/runs/exp106/weights-epoch600.nnue");

    std::cout << "Loaded network" << std::endl;

    DataSet data_to_write {};

    uint64_t batch_num = 0;
    while (batch_num++ < BatchesPerFile * TotalFiles) {
        printf("Running batch %lld\n", batch_num);

        auto* ds = batch_loader.next();

        Berserk::assign_inputs_batch(*ds, get<0>(inputs), get<1>(inputs), target, target_mask);

        get<0>(inputs).column_indices.gpu_upload();
        get<1>(inputs).column_indices.gpu_upload();
        target.gpu_upload();
        target_mask.gpu_upload();

        network->feed(vector<SparseInput*> {&get<0>(inputs), &get<1>(inputs)});

        auto* values = &network->getOutput().values;
        values->gpu_download();

        for (size_t i = 0; i < BatchSize; i++) {
            auto eval = (int16_t) round(values->get(i));
            auto* pos = &ds->positions[i];

            if (pos->m_meta.getActivePlayer() == BLACK)
                eval = -eval;

            pos->m_result.score = eval;
        }

        data_to_write.addData(*ds);

        if (batch_num % BatchesPerFile == 0) {
            int idx = batch_num / BatchesPerFile;

            write("E:/berserk/training-data/rescored/n5k." + to_string(idx) + ".bin", data_to_write);

            data_to_write.clear();
        }
    }

    // Trainer<Berserk, 600> trainer {};
    // trainer.fit(files, vector<string> {data_path + "validation.bin"}, output);

    close();
}
