#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_TensorList()
{
    LANTERN_FUNCTION_START
    return (void *)new LanternObject<std::vector<torch::Tensor>>();
    LANTERN_FUNCTION_END
}

void* _lantern_OptionalTensorList ()
{
    LANTERN_FUNCTION_START
    return (void*) new LanternObject<c10::List<c10::optional<torch::Tensor>>>();
    LANTERN_FUNCTION_END
}

void _lantern_TensorList_push_back(void *self, void *x)
{
    LANTERN_FUNCTION_START
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();
    reinterpret_cast<LanternObject<std::vector<torch::Tensor>> *>(self)->get().push_back(ten);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_OptionalTensorList_push_back (void* self, void* x, bool is_null)
{
    LANTERN_FUNCTION_START
    c10::optional<torch::Tensor> tensor;
    if (is_null)
    {
        tensor = c10::nullopt;
    }
    else
    {
        tensor = reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get();
    }
    reinterpret_cast<LanternObject<c10::List<c10::optional<torch::Tensor>>>*>(self)->get().push_back(tensor);
    LANTERN_FUNCTION_END_VOID
}

void *_lantern_TensorList_at(void *self, int64_t i)
{
    LANTERN_FUNCTION_START
    torch::Tensor out = reinterpret_cast<LanternObject<std::vector<torch::Tensor>> *>(self)->get().at(i);
    return (void *)new LanternObject<torch::Tensor>(out);
    LANTERN_FUNCTION_END
}

int64_t _lantern_TensorList_size(void *self)
{
    LANTERN_FUNCTION_START
    return reinterpret_cast<LanternObject<std::vector<torch::Tensor>> *>(self)->get().size();
    LANTERN_FUNCTION_END_RET(0)
}