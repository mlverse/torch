#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_Tensor_storage(void *self)
{
    LANTERN_FUNCTION_START
    torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    return (void *)new LanternObject<torch::Storage>(x.storage());
    LANTERN_FUNCTION_END
}

bool _lantern_Tensor_has_storage(void *self)
{
    LANTERN_FUNCTION_START
    torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    return x.has_storage();
    LANTERN_FUNCTION_END_RET(false)
}

const char *_lantern_Storage_data_ptr(void *self)
{
    LANTERN_FUNCTION_START
    torch::Storage x = reinterpret_cast<LanternObject<torch::Storage> *>(self)->get();
    std::stringstream ss;
    ss << x.data_ptr().get();
    std::string str = ss.str();
    char *cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    return cstr;
    LANTERN_FUNCTION_END
}
