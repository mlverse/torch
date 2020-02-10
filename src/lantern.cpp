#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

void lanternTest()
{
    std::cout << "-- Lantern: 0.1.0" << std::endl;

    std::cout << "-- Testing Tensor" << std::endl;
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    std::cout << "-- Success!" << std::endl;
}
