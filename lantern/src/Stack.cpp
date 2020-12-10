#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

void * _lantern_Stack_new ()
{
    LANTERN_FUNCTION_START;
    return (void*) new LanternObject<torch::jit::Stack>();
    LANTERN_FUNCTION_END;
}

int64_t _lantern_Stack_size (void* self)
{
    LANTERN_FUNCTION_START;
    return reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get().size();
    LANTERN_FUNCTION_END;
}

bool _lantern_Stack_at_is (void* self, int64_t i, const char * type)
{
    LANTERN_FUNCTION_START;
    auto type_ = std::string(type);
    auto el = reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get().at(i);
    
    if (type_ == "Tensor")
    {
        return el.isTensor();
    } else if (type_ == "Int")
    {
        return el.isInt();
    } else if (type_ == "TensorList")
    {
        return el.isTensorList();
    } else if (type_ == "List")
    {
        return el.isList();
    }
    else {
        throw std::runtime_error("Type '" + type_ + "' is not supported.");
    }

    LANTERN_FUNCTION_END;
}

void _lantern_Stack_push_back_Tensor (void* self, void * x)
{
    LANTERN_FUNCTION_START;
    torch::Tensor t = reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get();
    reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get().push_back(t);
    LANTERN_FUNCTION_END_VOID;
}

void _lantern_Stack_push_back_TensorList (void* self, void* x)
{
    LANTERN_FUNCTION_START;
    torch::TensorList tl = reinterpret_cast<LanternObject<std::vector<torch::Tensor>> *>(x)->get();
    reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get().push_back(tl);
    LANTERN_FUNCTION_END_VOID;
}

void _lantern_Stack_push_back_int64_t (void* self, int64_t x)
{
    LANTERN_FUNCTION_START;
    reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get().push_back(x);
    LANTERN_FUNCTION_END_VOID;
}

void * _lantern_Stack_at_Tensor (void* self, int64_t i)
{
    LANTERN_FUNCTION_START;
    auto s = reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get();
    auto t = s.at(i).toTensor();
    return (void*) new LanternObject<torch::Tensor>(t);
    LANTERN_FUNCTION_END;
}

void* _lantern_Stack_at_TensorList (void* self, int64_t i)
{
    LANTERN_FUNCTION_START;
    auto s = reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get();
    auto out = s.at(i).toTensorList();
    std::vector<torch::Tensor> o = out.vec();
    return (void*) new LanternObject<std::vector<torch::Tensor>>(o);
    LANTERN_FUNCTION_END;
}

int64_t _lantern_Stack_at_int64_t (void* self, int64_t i)
{
    LANTERN_FUNCTION_START;
    auto s = reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get();
    int64_t out = s.at(i).toInt();
    return out;
    LANTERN_FUNCTION_END;
}