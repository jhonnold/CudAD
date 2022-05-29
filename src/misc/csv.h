
//
// Created by Luecx on 18.12.2021.
//

#ifndef DIFFERENTIATION_SRC_MISC_CSV_H_
#define DIFFERENTIATION_SRC_MISC_CSV_H_

#include <cstdarg>
#include <fstream>
struct CSVWriter {

    std::ofstream csv_file {};

    CSVWriter(std::string res) { csv_file = std::ofstream {res}; }

    virtual ~CSVWriter() { csv_file.close(); }

    void write(std::initializer_list<std::string> args) {
        int i = 0;
        for (auto h : args) {
            if (i != 0)
                csv_file << ",";

            csv_file << "\"" << h << "\"";
            i++;
        }

        csv_file << std::endl;
    }
};

#endif    // DIFFERENTIATION_SRC_MISC_CSV_H_
