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

void* _lantern_Stack_at (void* self, int64_t index)
{
    auto self_ = reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get();
    return (void*) new c10::IValue(self_.at(index));
}

void _lantern_Stack_push_back_IValue (void* self, void* x)
{
    LANTERN_FUNCTION_START;
    reinterpret_cast<LanternObject<torch::jit::Stack> *>(self)->get().push_back(
        *reinterpret_cast<torch::IValue*>(x)
    );
    LANTERN_FUNCTION_END_VOID;
}