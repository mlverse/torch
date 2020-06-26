#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_Tensor_storage(void *self)
{
    torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    return (void *)new LanternObject<torch::Storage>(x.storage());
}

bool lantern_Tensor_has_storage(void *self)
{
    torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    return x.has_storage();
}

const char *lantern_Storage_data_ptr(void *self)
{
    torch::Storage x = reinterpret_cast<LanternObject<torch::Storage> *>(self)->get();
    std::stringstream ss;
    ss << x.data_ptr().get();
    std::string str = ss.str();
    char *cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    return cstr;
}
