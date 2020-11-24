#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

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



