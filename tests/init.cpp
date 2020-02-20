#include <iostream>
#include <fstream>

#include "lantern/lantern.h"

#include "init.h"

void myInit(char* path)
{
    std::string error;
    if (!lanternInit(path, &error)) {
        std::cout << "Error: " << error << std::endl;
        throw;
    }
}