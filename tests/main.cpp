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

    void* device = lantern_Device("cpu", 0, false);
    std::cout << "Device: " << lantern_Device_type(device) << ":" << lantern_Device_index(device) << std::endl;
    
    void* generator = lantern_Generator();
    lantern_Generator_set_current_seed(generator, 123456);
    std::cout << "Seed: " << lantern_Generator_current_seed(generator) << std::endl;
    
    void* qscheme = lantern_QScheme_per_channel_affine();
    std::cout << "QScheme: " << lantern_QScheme_type(qscheme) << std::endl;

    return 0;
}