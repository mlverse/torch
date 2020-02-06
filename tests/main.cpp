#include <iostream>
#include "lantern/lantern.h"

int main(int argc, char *argv[])
{
    std::string error;
    if (!lanternInit(argv[1], &error)) {
        std::cout << "Error: " << error;
        return 1;
    }

    lanternPrint();

    return 0;
}