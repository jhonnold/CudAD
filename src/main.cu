#include "activations/ClippedReLU.h"
#include "activations/Linear.h"
#include "activations/ReLU.h"
#include "activations/Sigmoid.h"
#include "data/DenseMatrix.h"
#include "data/SArray.h"
#include "data/Tape.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "dataset/io.h"
#include "dataset/reader.h"
#include "dataset/writer.h"
#include "dataset/shuffle.h"
#include "layer/DenseLayer.h"
#include "layer/DuplicateDenseLayer.h"
#include "loss/MLE.h"
#include "loss/MPE.h"
#include "loss/MSE.h"
#include "mappings.h"
#include "misc/csv.h"
#include "misc/timer.h"
#include "network/Network.h"
#include "operations/operations.h"
#include "optimizer/Adam.h"
#include "position/fenparsing.h"
#include "position/position.h"
#include "quantitize.h"

#include <filesystem>
#include <iostream>

const std::string data_path = "E:/berserk/training-data/berserk9dev2/finny-data/";
std::string output = "./resources/runs/exp4/";

int main() {
    init();

    // definitions
    constexpr uint32_t       I = 8 * 12 * 64;
    constexpr uint32_t      L1 = 256;
    constexpr uint32_t      L2 = 32;
    constexpr uint32_t      L3 = 32;  
    constexpr uint32_t       O = 1;
    constexpr uint32_t       B = 8192;
    constexpr uint32_t     BPE = 100000000 / B;
    constexpr  int32_t       E = 600;

    // Load files
    std::vector<std::string> files {};
    for (int i = 0; i < 7; i++)
        files.push_back(data_path + "berserk9dev2.d9." + std::to_string(i) + ".bin");

    BatchLoader  batch_loader {files, B};

    // Input data (perspective)
    SparseInput  i0 {I, B, 32};    // 32 max inputs
    SparseInput  i1 {I, B, 32};

    DenseMatrix  target {O, B};
    SArray<bool> target_mask {O * B};
    target_mask.malloc_cpu();
    target_mask.malloc_gpu();

    const float QUANT_ONE = 127.0;
    DuplicateDenseLayer<I, L1, ClippedReLU> l1 {};
    dynamic_cast<ClippedReLU*>(l1.getActivationFunction())->max = 1.0;

    const float SCALE_HIDDEN = 64.0;
    DenseLayer<2 * L1, L2, ClippedReLU> l2 {};
    dynamic_cast<ClippedReLU*>(l2.getActivationFunction())->max = 1.0;
    l2.getTunableParameters()[0]->min_allowed_value = -QUANT_ONE / SCALE_HIDDEN;
    l2.getTunableParameters()[0]->max_allowed_value = QUANT_ONE / SCALE_HIDDEN;

    DenseLayer<L2, L3, ClippedReLU> l3 {};
    dynamic_cast<ClippedReLU*>(l3.getActivationFunction())->max = 1.0;
    l3.getTunableParameters()[0]->min_allowed_value = -QUANT_ONE / SCALE_HIDDEN;
    l3.getTunableParameters()[0]->max_allowed_value = QUANT_ONE / SCALE_HIDDEN;

    const float SCALE_OUT = 16.0;
    const float NN_SCALE = 231.0;
    DenseLayer<L3, O, Sigmoid> l4 {};
    dynamic_cast<Sigmoid*>(l4.getActivationFunction())->scalar = NN_SCALE / 139;
    l4.getTunableParameters()[0]->min_allowed_value = -(QUANT_ONE * QUANT_ONE) / (SCALE_OUT * NN_SCALE);
    l4.getTunableParameters()[0]->max_allowed_value = (QUANT_ONE * QUANT_ONE) / (SCALE_OUT * NN_SCALE);

    // stack layers to build network
    std::vector<LayerInterface*> layers {};
    layers.push_back(&l1);
    layers.push_back(&l2);
    layers.push_back(&l3);
    layers.push_back(&l4);

    Network network {layers};

    // loss function
    MPE     loss_function {2.5, true};
    network.setLossFunction(&loss_function);

    // optimizer
    Adam adam {};
    adam.init(layers);
    adam.alpha = 0.001;
    adam.beta1 = 0.9;
    adam.beta2 = 0.999;
    adam.eps = 1e-7;

    CSVWriter csv {output + "loss.csv"};

    Timer t {};
    for (int epoch = 1; epoch <= E; epoch++) {
        float epoch_loss = 0;
        long long prev_duration = 0;

        t.tick();

        for (int batch = 1; batch <= BPE; batch++) {
            // get the next dataset (batch)
            auto* ds = batch_loader.next();
            // assign to the inputs and compute the target
            dense_berky::assign_inputs_batch(*ds, i0, i1, target, target_mask);
            // upload relevant data
            i0.column_indices.gpu_upload();
            i1.column_indices.gpu_upload();
            target.gpu_upload();
            target_mask.gpu_upload();

            // download the loss to display the loss of the iteration
            loss_function.loss.gpu_download();

            // measure time and print output
            t.tock();
            if (batch == BPE || t.duration() - prev_duration > 1000) {
                prev_duration = t.duration();

                std::printf("\rep/ba = [%3d/%5d], ", epoch, batch + 1);
                std::printf("batch_loss = [%1.8f], ", loss_function.loss(0));
                std::printf("epoch_loss = [%1.8f], ", epoch_loss / (batch + 1));
                std::printf("speed = [%9d pos/s], ", (int) std::round(1000.0f * B * (batch + 1) / t.duration()));
                std::printf("time = [%3ds]", (int) t.duration() / 1000);
                std::cout << std::flush;
            }

            epoch_loss += loss_function.loss(0);
            // make sure to reset the loss here since the mse increments the loss in order to not have
            // to use memcpy (might change soon)
            loss_function.loss(0) = 0;
            loss_function.loss.gpu_upload();

            // feed forward
            network.batch(std::vector<SparseInput*> {&i0, &i1}, target, target_mask);

            // update weights
            adam.apply(1);
        }

        std::cout << std::endl;

        csv.write({std::to_string(epoch),  std::to_string(epoch_loss / BPE)});
        write_4(output + "nn-epoch" + std::to_string(epoch) + ".nnue", network, QUANT_ONE, SCALE_HIDDEN, SCALE_OUT, NN_SCALE);
        
        if (epoch % 100 == 0)
            adam.alpha *= 0.3;
    }

    close();
}
