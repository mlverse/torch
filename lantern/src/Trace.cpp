#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

void * _lantern_CompilationUnit_new ()
{
    LANTERN_FUNCTION_START;
    return (void*) new LanternObject<torch::jit::CompilationUnit>();
    LANTERN_FUNCTION_END;
}

void* _lantern_create_traceable_fun (void* fn)
{
    LANTERN_FUNCTION_START;
    auto r_fn = *reinterpret_cast<std::function<void*(void*)>*>(fn);
    std::function<Stack(Stack)> tr_fn = [r_fn](Stack x)
    {
        auto tmp = LanternObject<Stack>(x);
        void* out = r_fn((void*) &x);
        return reinterpret_cast<LanternObject<Stack>*>(out)->get();
    };

    return (void*) new LanternObject<std::function<Stack(Stack)>>(tr_fn); 
    LANTERN_FUNCTION_END;
}

int _lantern_trace_fn (void* fn, void* inputs)
{
    LANTERN_FUNCTION_START;
    std::function<Stack(Stack)> fn_ = reinterpret_cast<LanternObject<std::function<Stack(Stack)>>*>(fn)->get();
    Stack inputs_ = reinterpret_cast<LanternObject<Stack>*>(inputs)->get();

    std::function<std::string(const torch::autograd::Variable&)> var_fn = [](const torch::autograd::Variable& x) {
        return "";
    };

    std::cout << "Start tracing!" << std::endl;

    auto traced = torch::jit::tracer::trace(
        inputs_,
        fn_,
        var_fn
    );

    std::cout << "Tracing worked!" << std::endl;

    auto cu = torch::jit::CompilationUnit();
    auto z = cu.create_function("name", std::get<0>(traced)->graph, true);

    std::cout << (*z)(inputs_).toTensor() << std::endl;

    return 0;
    LANTERN_FUNCTION_END;
}

void _trace_r_nn_module ()
{
    LANTERN_FUNCTION_START;
    auto x = torch::randn({10, 10});
    
    torch::jit::Stack inputs;
    inputs.push_back(x);

    std::function<torch::jit::Stack(torch::jit::Stack)> fn = [](torch::jit::Stack x) {
        auto res = torch::relu(x[0].toTensor());
        res = torch::exp(res);
        torch::jit::Stack output;
        output.push_back(res);
        return output;
    };

    std::function<std::string(const torch::autograd::Variable&)> var_fn = [](const torch::autograd::Variable& x) {
        return "";
    };

    auto traced = torch::jit::tracer::trace(
        inputs,
        fn,
        var_fn
    );

    auto cu = torch::jit::CompilationUnit();
    auto z = cu.create_function("name", std::get<0>(traced)->graph, true);
    
    std::cout << "calling JIT" << std::endl;

    auto t = std::get<0>(traced);
    std::cout << t->graph->toString() << std::endl;
    auto node = t->graph->param_node();
    std::cout << node->attributeNames() << std::endl;



    std::cout << "calling JIT" << std::endl;
    std::cout << (*z)(inputs).toTensor() << std::endl;
    LANTERN_FUNCTION_END_VOID;
}



