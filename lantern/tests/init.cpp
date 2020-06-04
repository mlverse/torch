#include <iostream>
#include <fstream>

#include "lantern/lantern.h"

#include "init.h"

void myInit(char* path)
{
    std::string error;
    std::cout << "Lantern path: " << std::string(path) << std::endl;
    if (!lanternInit(path, &error)) {
        std::cout << "Error: " << error << std::endl;
        throw;
    }
}