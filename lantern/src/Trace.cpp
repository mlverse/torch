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
    return (void*) new torch::jit::CompilationUnit();
    LANTERN_FUNCTION_END;
}

void* _lantern_create_traceable_fun (void* fn)
{
    LANTERN_FUNCTION_START;
    std::function<Stack(Stack)> tr_fn = [fn](Stack x)
    {
        auto r_fn = *reinterpret_cast<std::function<void*(void*)>*>(fn);
        auto tmp = new LanternObject<Stack>(x);
        void* out = r_fn((void*) tmp);
        return reinterpret_cast<LanternObject<Stack>*>(out)->get();
    };

    return (void*) new LanternObject<std::function<Stack(Stack)>>(tr_fn); 
    LANTERN_FUNCTION_END;
}

void* _lantern_trace_fn (void* fn, void* inputs, void* compilation_unit)
{
    LANTERN_FUNCTION_START;
    std::function<Stack(Stack)> fn_ = reinterpret_cast<LanternObject<std::function<Stack(Stack)>>*>(fn)->get();
    Stack inputs_ = reinterpret_cast<LanternObject<Stack>*>(inputs)->get();
    CompilationUnit* cu = reinterpret_cast<CompilationUnit*>(compilation_unit);

    std::function<std::string(const torch::autograd::Variable&)> var_fn = [](const torch::autograd::Variable& x) {
        return "";
    };

    auto traced = torch::jit::tracer::trace(
        inputs_,
        fn_,
        var_fn
    );

    auto tr_fn = cu->create_function("name", std::get<0>(traced)->graph, true);
    
    return (void*) tr_fn;
    LANTERN_FUNCTION_END;
}

void* _lantern_call_traced_fn (void* fn, void* inputs)
{
  Function* fn_ = reinterpret_cast<Function *>(fn);
  Stack inputs_ = reinterpret_cast<LanternObject<Stack>*>(inputs)->get();

  auto outputs = new LanternObject<torch::jit::Stack>();
  auto out = (*fn_)(inputs_);
  outputs->get().push_back(out);  
  
  return (void*) outputs;
}

void addFunctionToModule(Module& module, const Function* func) {
  // Make a graph with a fake self argument
  auto graph = func->graph()->copy();
  auto v = graph->insertInput(0, "self");
  v->setType(module._ivalue()->type());
  const auto name = QualifiedName(*module.type()->name(), "forward");
  auto method =
      module._ivalue()->compilation_unit()->create_function(name, graph);
  module.type()->addMethod(method);
}

void _lantern_traced_fn_save (void* fn, const char* filename)
{
    LANTERN_FUNCTION_START;
    Function* fn_ = reinterpret_cast<Function *>(fn);
    auto filename_ = std::string(filename);
    
    Module module("__torch__.PlaceholderModule");
    
    module.register_attribute("training", BoolType::get(), true);
    addFunctionToModule(module, fn_);
    module.save(filename_);
    LANTERN_FUNCTION_END_VOID;
}

const char * _lantern_traced_fn_graph_print (void * fn)
{
  LANTERN_FUNCTION_START
  Function* fn_ = reinterpret_cast<Function *>(fn);
  std::string str = fn_->graph()->toString();
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
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



