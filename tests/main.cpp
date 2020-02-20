#include <iostream>
#include <fstream>

#define LANTERN_HEADERS_ONLY
#include "lantern/lantern.h"

#include "init.h"

int main(int argc, char *argv[])
{
    if (argc <= 1) {
        std::cout << "Usage: lanterntest <path-to-lib>";
        return 1;
    }

    myInit(argv[1]);
    lanternTest();

    return 0;
}