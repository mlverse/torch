#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"
void *_lantern_Layout_strided()
{
    LANTERN_FUNCTION_START
    return (void *)new LanternObject<torch::Layout>(torch::kStrided);
    LANTERN_FUNCTION_END
}

void *_lantern_Layout_sparse()
{
    LANTERN_FUNCTION_START
    return (void *)new LanternObject<torch::Layout>(torch::kSparse);
    LANTERN_FUNCTION_END
}

const char *_lantern_Layout_string(void *x)
{
    LANTERN_FUNCTION_START
    auto out = new std::string;
    auto l = reinterpret_cast<LanternObject<torch::Layout> *>(x)->get();
    if (l == torch::kStrided)
    {
        *out = "strided";
    }
    else if (l == torch::kSparse)
    {
        *out = "sparse_coo";
    }
    return out->c_str();
    LANTERN_FUNCTION_END
}