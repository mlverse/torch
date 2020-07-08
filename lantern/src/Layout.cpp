#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"
void *_lantern_Layout_strided()
{
    return (void *)new LanternObject<torch::Layout>(torch::kStrided);
}

void *_lantern_Layout_sparse()
{
    return (void *)new LanternObject<torch::Layout>(torch::kSparse);
}

const char *_lantern_Layout_string(void *x)
{
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
}