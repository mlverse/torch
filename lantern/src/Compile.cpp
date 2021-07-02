#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h> // One-stop header.
#include "utils.hpp"

void* _lantern_jit_compile (void* source, void* cu)
{
    LANTERN_FUNCTION_START
    const auto source_ = *reinterpret_cast<std::string*>(source);
    auto result = new torch::jit::CompilationUnit(std::move(*torch::jit::compile(source_)));
    return (void*) result;
    LANTERN_FUNCTION_END
}

void* _lantern_jit_compile_list_methods (void* cu)
{
    LANTERN_FUNCTION_START
    auto cu_ = reinterpret_cast<torch::jit::CompilationUnit*>(cu);
    auto funs = cu_->get_functions();
    std::vector<std::string> names;
    for (const auto& f: funs)
    {
        names.push_back(f->name());
    }
    return (void*) new std::vector<std::string>(names);
    LANTERN_FUNCTION_END
}

void* _lantern_jit_compile_get_method (void* cu, void* name)
{
    LANTERN_FUNCTION_START
    auto cu_ = reinterpret_cast<torch::jit::CompilationUnit*>(cu);
    auto name_ = *reinterpret_cast<std::string*>(name);
    return (void*) cu_->find_function(name_);
    LANTERN_FUNCTION_END
}