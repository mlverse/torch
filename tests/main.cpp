#include <iostream>
#include <fstream>

#include "lantern/lantern.h"

inline bool exists(const std::string& path) {
    if (FILE *file = fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

int main(int argc, char *argv[])
{
    if (argc < 2 || !exists(argv[1])) {
        std::cout << "Error: Incorrect path provided";
        return 1;
    }

    std::string error;
    if (!lanternInit(argv[1], &error)) {
        std::cout << "Error: " << error;
        return 1;
    }

    lanternTest();

    return 0;
}