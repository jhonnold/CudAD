
//
// Created by Luecx on 18.12.2021.
//

#ifndef DIFFERENTIATION_SRC_MISC_CSV_H_
#define DIFFERENTIATION_SRC_MISC_CSV_H_

#include <cstdarg>
#include <fstream>
struct CSVWriter {

    std::ofstream csv_file{};

    CSVWriter(std::string res){
        csv_file = std::ofstream {res};
    }

    virtual ~CSVWriter() {
        csv_file.close();
    }

    void write(std::initializer_list<std::string> args) {
        csv << "\"" << args[0] << "\"";

        for (size_t i = 1; i < args.size(); i++)
            csv_file << "," << "\"" << args[i] << "\"";

        csv_file << std::endl;
    }

};

#endif    // DIFFERENTIATION_SRC_MISC_CSV_H_
