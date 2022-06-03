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

const std::string data_path = "E:/berserk/training-data/n5k/";
std::string output = "./resources/runs/exp18/";

float validate(Network&     network,
               DataSet&     data_set,
               DenseMatrix& target,
               SArray<bool>& target_mask,
               SparseInput& i1,
               SparseInput& i2);

int main() {
    init();

    // definitions
    constexpr uint32_t       I = 8 * 12 * 64;
    constexpr uint32_t       H = 512;
    constexpr uint32_t      L2 = 8;
    constexpr uint32_t       O = 1;
    constexpr uint32_t       B = 16384;
    constexpr uint32_t     BPE = 100000000 / B;
    constexpr  int32_t       E = 450;

    // Load files
    std::vector<std::string> files {};
    for (int i = 0; i < 20; i++)
        files.push_back(data_path + "n5k." + std::to_string(i) + ".bin");

    BatchLoader  batch_loader {files, B};
    DataSet validation = read<BINARY>(data_path + "validation.bin");

    // Input data (perspective)
    SparseInput  i0 {I, B, 32};    // 32 max inputs
    SparseInput  i1 {I, B, 32};

    DenseMatrix  target {O, B};
    SArray<bool> target_mask {O * B};
    target_mask.malloc_cpu();
    target_mask.malloc_gpu();

    // 1536 -> (2x512) -> 1
    DuplicateDenseLayer<I, H, ReLU> l1 {};
    l1.lasso_regularization = 1.0 / 8388608.0;

    DenseLayer<H * 2, L2, ReLU> l2 {};

    DenseLayer<L2, O, Sigmoid>   l3 {};
    dynamic_cast<Sigmoid*>(l3.getActivationFunction())->scalar = 1.0 / 139;

    // stack layers to build network
    std::vector<LayerInterface*> layers {};
    layers.push_back(&l1);
    layers.push_back(&l2);
    layers.push_back(&l3);

    Network network {layers};

    // loss function
    MPE     loss_function {2.5, false};
    network.setLossFunction(&loss_function);

    // optimizer
    Adam adam {};
    adam.init(layers);
    adam.alpha = 0.01;
    adam.beta1 = 0.95;
    adam.beta2 = 0.999;

    CSVWriter csv {output + "loss.csv"};
    csv.write({"epoch", "training_loss", "validation_loss"});

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

        float validation_loss = validate(network, validation, target, target_mask, i0, i1);
        t.tock();
        std::printf("\rep/ba = [%3d/%5d], ", epoch, BPE);
        std::printf("valid_loss = [%1.8f], ", validation_loss);
        std::printf("epoch_loss = [%1.8f], ", epoch_loss / BPE);
        std::printf("speed = [%9d pos/s], ", (int) std::round(1000.0f * (B * BPE + validation.header.position_count) / t.duration()));
        std::printf("time = [%3ds]", (int) t.duration() / 1000);
        std::cout << std::endl;

        csv.write({std::to_string(epoch),  std::to_string(epoch_loss / BPE), std::to_string(validation_loss)});

        // computeScalars(batch_loader, network, 128, I);

        if (epoch % 10 == 0)
            write_3(output + "nn-epoch" + std::to_string(epoch) + ".nnue", network);

        if (epoch % 100 == 0)
            adam.alpha *= 0.3;
    }

    close();
}

float validate(Network&     network,
               DataSet&     data_set,
               DenseMatrix& target,
               SArray<bool>& target_mask,
               SparseInput& i1,
               SparseInput& i2) {

    int B = i1.n;

    // reset loss
    float prev_loss = network.getLossFunction()->getLoss().get(0);
    network.getLossFunction()->getLoss().get(0) = 0;
    network.getLossFunction()->getLoss().gpu_upload();

    int c = std::floor(data_set.positions.size() / B);
    for(int i = 0; i < c; i++){
        int id1 = i   * B;
        int id2 = id1 + B;
        DataSet temp{};
        temp.header.position_count = B;
        temp.positions.assign(&data_set.positions[id1],&data_set.positions[id2]);

        dense_berky::assign_inputs_batch(temp, i1, i2, target, target_mask);

        i1.column_indices.gpu_upload();
        i2.column_indices.gpu_upload();
        target.gpu_upload();
        target_mask.gpu_upload();

        network.feed(std::vector<SparseInput*> {&i1, &i2});

        network.getLossFunction()->apply(network.getOutput().values,
                                         network.getOutput().gradients,
                                         target,
                                         target_mask,
                                         DEVICE);
    }

    network.getLossFunction()->getLoss().gpu_download();
    float loss = network.getLossFunction()->getLoss().get(0);

    network.getLossFunction()->getLoss().get(0) = prev_loss;
    network.getLossFunction()->getLoss().gpu_upload();

    return loss / c;
}
