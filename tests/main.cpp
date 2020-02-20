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

    void* device = lanternDevice("cpu", 0, false);
    std::cout << "Device: " << lanternDeviceType(device) << ":" << lanternDeviceIndex(device) << std::endl;

    return 0;
}