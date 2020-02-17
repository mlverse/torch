#include <iostream>
#include <fstream>

#include "lantern/lantern.h"

int main(int argc, char *argv[])
{
    if (argc <= 1) {
        std::cout << "Usage: lanterntest <path-to-lib>";
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