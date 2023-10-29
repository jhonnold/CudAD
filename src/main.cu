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
#include "dataset/reader.h"
#include "dataset/shuffle.h"
#include "dataset/writer.h"
#include "misc/config.h"
#include "position/zobrist.h"
#include "trainer.h"

#include <iostream>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

int main() {
    init();

    vector<string> files {};

    for (const auto& entry : fs::directory_iterator("C:/Programming/berserk-data/bins")) {
        const std::string path = entry.path().string();
        files.push_back(path);

        std::cout << "Adding file " << path << std::endl;
    }

    mix_and_shuffle_2(files, "C:/Programming/berserk-data/data212/data212.$.bin", 200);

    // size_t t = 1;
    // for (const auto& entry : fs::directory_iterator("E:/berserk-fen-gen/dfrc-min5k")) {
    //     auto file = entry.path().string();
    //     std::cout << "Reading from " << file << std::endl;
    //     auto ds = read<TEXT>(file);

    //     auto outfile = "C:/Programming/berserk-data/bins/berserk20230116b.dfrc." + to_string(t++) + ".bin";
    //     std::cout << "Writing to " << outfile << std::endl;
    //     write(outfile, ds);
    // }

    // for (const auto& entry : fs::directory_iterator("E:/berserk-fen-gen/spot2")) {
    //     auto file = entry.path().string();
    //     std::cout << "Reading from " << file << std::endl;
    //     auto ds = read<TEXT>(file);

    //     auto outfile = "C:/Programming/berserk-data/bins/berserk20231016.20k." + to_string(t++) + ".bin";
    //     std::cout << "Writing to " << outfile << std::endl;
    //     write(outfile, ds);
    // }

    // const string data_path = "C:/Programming/berserk-data/data206/";
    // const string output    = "./resources/runs/exp203/";

    // // Load files
    // vector<string> files {};
    // for (int i = 1; i <= 200; i++)
    //     files.push_back(data_path + "data206." + to_string(i) + ".bin");

    // size_t      total      = 2e9;
    // size_t      chunks     = 20;
    // size_t      batch_size = 10000;

    // BatchLoader loader {files, 10000};
    // loader.start();

    // for (size_t i = 1; i <= chunks; i++) {
    //     std::string fname       = "E:/zuppa/zuppa." + to_string(i) + ".txt";

    //     std::cout << "Writing to " << fname << std::endl;

    //     FILE*       fout        = fopen(fname.c_str(), "w");

    //     size_t      inner_total = 0;
    //     while (inner_total < total / chunks) {
    //         inner_total += batch_size;

    //         auto ds = loader.next();
    //         for (Position pos : ds->positions) {
    //             fputs(writeFen(pos, true).c_str(), fout);
    //             fputs("\n", fout);
    //         }

    //         if (inner_total % (total / chunks / 100) == 0)
    //             std::cout << inner_total << std::endl;
    //     }

    //     fclose(fout);
    // }

    // Trainer<Berserk, 1500> trainer {};
    // trainer.fit(files, vector<string> {data_path + "validation.bin"}, output);

    close();
}
