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
#include "dataset/writer.h"

#include <iostream>
#include <vector>

using namespace std;

int main() {
    init();

    const string data_path = "C:/Programming/berserk-data/exp203/";
    const string output    = "./resources/runs/exp203/";

    // Load files
    vector<string> files {};
    for (int i = 1; i <= 200; i++)
        files.push_back(data_path + "exp203." + to_string(i) + ".bin");

    Trainer<Berserk, 1500> trainer {};
    trainer.fit(files, vector<string> {data_path + "validation.bin"}, output);

    close();
}
