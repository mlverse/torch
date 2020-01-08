#include "torch_types.h"

torch::Device device_from_string(std::string device);

std::string device_to_string (torch::Device x);
