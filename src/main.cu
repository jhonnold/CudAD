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
#include "dataset/shuffle.h"
#include "trainer.h"
#include "position/zobrist.h"
#include "dataset/reader.h"

#include <iostream>
#include <vector>

using namespace std;

// static constexpr uint64_t HASH_SIZE = 1024 * 1024 * 1024;

int main() {
    init();

    // PRNG rng(1070372);
    // Zobrist zobrist(rng);

    // vector<Key> hash;
    // hash.resize(HASH_SIZE);

    // uint64_t duplicates = 0;

    // vector<string> files {};
    // for (int i = 1; i <= 100; i++) {
    //     auto ds = read<BINARY>("E:/berserk/training-data/exp134/exp134." + to_string(i) + ".bin");
    //     DataSet output {};

    //     for (Position pos : ds.positions) {
    //         Key key = zobrist.get_key(pos);
            
    //         size_t idx = (size_t)(key & (HASH_SIZE - 1));
    //         Key old = hash[idx];

    //         if (key == old) {
    //             duplicates++;
    //         } else {
    //             hash[idx] = key;
    //             output.positions.push_back(pos);
    //         }
    //     }
        
    //     cout << "Duplicates: " << duplicates << endl;

    //     output.header.position_count = output.positions.size();
    //     write("E:/berserk/training-data/exp135/exp135." + to_string(i) + ".bin", output);
    // }

    const string data_path = "E:/berserk/training-data/exp135/";
    const string output    = "./resources/runs/exp135/";

    // Load files
    vector<string> files {};
    for (int i = 1; i <= 100; i++)
        files.push_back(data_path + "exp135." + to_string(i) + ".bin");

    Trainer<Berserk, 600> trainer {};
    trainer.fit(files, vector<string> {data_path + "validation.bin"}, output);

    close();
}
