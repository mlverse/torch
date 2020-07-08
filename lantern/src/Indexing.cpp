#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

using namespace torch::indexing;

void *_lantern_TensorIndex_new()
{
    LANTERN_FUNCTION_START
    return (void *)new LanternObject<std::vector<at::indexing::TensorIndex>>();
    LANTERN_FUNCTION_END
}

void _lantern_TensorIndex_append_tensor(void *self, void *x)
{
    LANTERN_FUNCTION_START
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(ten);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_ellipsis(void *self)
{
    LANTERN_FUNCTION_START
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(Ellipsis);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_slice(void *self, void *x)
{
    LANTERN_FUNCTION_START
    torch::indexing::Slice slice = reinterpret_cast<LanternObject<torch::indexing::Slice> *>(x)->get();
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(slice);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_none(void *self)
{
    LANTERN_FUNCTION_START
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(None);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_bool(void *self, bool x)
{
    LANTERN_FUNCTION_START
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(x);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_int64(void *self, int64_t x)
{
    LANTERN_FUNCTION_START
    reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(self)->get().push_back(x);
    LANTERN_FUNCTION_END_VOID
}

void *_lantern_Slice(void *start, void *end, void *step)
{
    LANTERN_FUNCTION_START
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
    LANTERN_FUNCTION_END
}

void *_lantern_Tensor_index(void *self, void *index)
{
    LANTERN_FUNCTION_START
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    torch::Tensor out = ten.index(reinterpret_cast<LanternObject<std::vector<at::indexing::TensorIndex>> *>(index)->get());
    return (void *)new LanternObject<torch::Tensor>(out);
    LANTERN_FUNCTION_END
}
