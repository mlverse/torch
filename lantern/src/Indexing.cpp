#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

using namespace torch::indexing;

void *lantern_TensorIndex_new()
{
    return (void *)new LanternObject<std::vector<at::indexing::TensorIndex>>();
}

void lantern_TensorIndex_append_tensor(void *self, void *x)
{
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(ten);
}

void lantern_TensorIndex_append_ellipsis(void *self)
{
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(Ellipsis);
}

void lantern_TensorIndex_append_slice(void *self, void *x)
{
    torch::indexing::Slice slice = reinterpret_cast<LanternObject<torch::indexing::Slice> *>(x)->get();
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(slice);
}

void lantern_TensorIndex_append_none(void *self)
{
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(None);
}

void lantern_TensorIndex_append_bool(void *self, bool x)
{
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(x);
}

void *lantern_Tensor_index(void *self, void *index)
{
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    torch::Tensor out = ten.index(reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(index)->get());
    return (void *)new LanternObject<torch::Tensor>(out);
}
