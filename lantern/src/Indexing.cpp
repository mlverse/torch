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

void lantern_TensorIndex_append_int64(void *self, int64_t x)
{
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(x);
}

void *lantern_Slice(void *start, void *end, void *step)
{
    auto start_ = reinterpret_cast<LanternObject<c10::optional<int64_t>> *>(start)->get();
    if (start_ == c10::nullopt)
    {
        start_ = None;
    }

    auto end_ = reinterpret_cast<LanternObject<c10::optional<int64_t>> *>(end)->get();
    if (end_ == c10::nullopt)
    {
        end_ = None;
    }

    auto step_ = reinterpret_cast<LanternObject<c10::optional<int64_t>> *>(step)->get();
    if (step_ == c10::nullopt)
    {
        step_ = None;
    }

    auto out = torch::indexing::Slice(start_, end_, step_);
    return (void *)new LanternObject<Slice>(out);
}

void *lantern_Tensor_index(void *self, void *index)
{
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    torch::Tensor out = ten.index(reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(index)->get());
    return (void *)new LanternObject<torch::Tensor>(out);
}

void lantern_Tensor_index_put_tensor_ (void* self, void* index, void* rhs)
{
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    auto i = reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(index)->get();
    torch::Tensor r = reinterpret_cast<LanternObject<torch::Tensor> *>(rhs)->get();
    ten.index_put_(i, r);
}

void lantern_Tensor_index_put_scalar_ (void* self, void* index, void* rhs)
{
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    auto i = reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(index)->get();
    torch::Scalar r = reinterpret_cast<LanternObject<torch::Scalar> *>(rhs)->get();
    ten.index_put_(i, r);
}

