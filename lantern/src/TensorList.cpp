#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_TensorList()
{
    LANTERN_FUNCTION_START
    return make_unique::TensorList({});
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
    torch::Tensor ten = from_raw::Tensor(x);
    from_raw::TensorList(self).push_back(ten);
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
        tensor = from_raw::Tensor(x);
    }
    reinterpret_cast<LanternObject<c10::List<c10::optional<torch::Tensor>>>*>(self)->get().push_back(tensor);
    LANTERN_FUNCTION_END_VOID
}

int64_t _lantern_OptionalTensorList_size (void* self)
{
    LANTERN_FUNCTION_START
    return reinterpret_cast<LanternObject<c10::List<c10::optional<torch::Tensor>>>*>(self)->get().size();
    LANTERN_FUNCTION_END
}

void* _lantern_OptionalTensorList_at (void* self, int64_t i)
{
    LANTERN_FUNCTION_START
    auto t = reinterpret_cast<LanternObject<c10::List<c10::optional<torch::Tensor>>>*>(self)->get().get(i);
    return make_unique::Tensor(t.value());
    LANTERN_FUNCTION_END
}

bool _lantern_OptionalTensorList_at_is_null (void* self, int64_t i)
{
    LANTERN_FUNCTION_START
    return (!reinterpret_cast<LanternObject<c10::List<c10::optional<torch::Tensor>>>*>(self)->get().get(i).has_value());
    LANTERN_FUNCTION_END
}

void* _lantern_TensorList_at(void *self, int64_t i)
{
    LANTERN_FUNCTION_START
    torch::Tensor out = from_raw::TensorList(self).at(i);
    return make_unique::Tensor(out);
    LANTERN_FUNCTION_END
}

int64_t _lantern_TensorList_size(void *self)
{
    LANTERN_FUNCTION_START
    return from_raw::TensorList(self).size();
    LANTERN_FUNCTION_END_RET(0)
}

void* _lantern_Stream ()
{
    c10::Stream x = c10::Stream(c10::Stream::Default(),torch::Device(torch::DeviceType::CPU));
    return (void*) new LanternObject<c10::Stream>(x);
}