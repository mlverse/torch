# Context-manager that enable anomaly detection for the autograd engine.

This does two things:

## Usage

``` r
with_detect_anomaly(code)
```

## Arguments

- code:

  Code that will be executed in the detect anomaly context.

## Details

- Running the forward pass with detection enabled will allow the
  backward pass to print the traceback of the forward operation that
  created the failing backward function.

- Any backward computation that generate "nan" value will raise an
  error.

## Warning

This mode should be enabled only for debugging as the different tests
will slow down your program execution.

## Examples

``` r
if (torch_is_installed()) {
x <- torch_randn(2, requires_grad = TRUE)
y <- torch_randn(1)
b <- (x^y)$sum()
y$add_(1)

try({
  b$backward()

  with_detect_anomaly({
    b$backward()
  })
})
}
#> Error in (function (self, inputs, gradient, retain_graph, create_graph)  : 
#>   one of the variables needed for gradient computation has been modified by an inplace operation: [CPUFloatType [1]] is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
#> Exception raised from unpack at /Users/runner/work/libtorch-mac-m1/libtorch-mac-m1/pytorch/torch/csrc/autograd/saved_variable.cpp:193 (most recent call first):
#> frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 52 (0x105ed455c in libc10.dylib)
#> frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) + 140 (0x105ed11ac in libc10.dylib)
#> frame #2: torch::autograd::SavedVariable::unpack(std::__1::shared_ptr<torch::autograd::Node>) const + 2380 (0x11a08f8e4 in libtorch_cpu.dylib)
#> frame #3: torch::autograd::generated::PowBackward1::apply(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>>&&) + 68 (0x118ce6000 in libtorch_cpu.dylib)
#> frame #4: torch::autograd::Node::operator()(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>>&&) + 116 (0x11a05a95c in libtorch_cpu.dylib)
#> frame #5: torch::autograd::Engine::evaluate_function(std::__1::shared_ptr<torch::autograd::GraphTask>&, torch::autograd::Node*, torch::autograd::InputBuffer&, std::__1::shared_ptr<torch::autograd::ReadyQueue> const&) + 2808 (0x11a0529c0 in libtorch_cpu.dylib)
#> frame #6: torch::autograd::Engine::thread_main(std::__1::shared_ptr<torch::autograd::GraphTask> const&) + 900 (0x11a05190c in libtorch_cpu.dylib)
#> frame #7: torch::autograd::Engine::execute_with_graph_task(std::__1::shared_ptr<torch::autograd::GraphTask> const&, std::__1::shared_ptr<torch::autograd::Node>, torch::autograd::InputBuffer&&) + 468 (0x11a0598cc in libtorch_cpu.dylib)
#> frame #8: torch::autograd::Engine::execute(std::__1::vector<torch::autograd::Edge, std::__1::allocator<torch::autograd::Edge>> const&, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, bool, bool, bool, std::__1::vector<torch::autograd::Edge, std::__1::allocator<torch::autograd::Edge>> const&) + 1936 (0x11a0585d8 in libtorch_cpu.dylib)
#> frame #9: torch::autograd::run_backward(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, bool, bool, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, bool, bool) + 836 (0x11a03f588 in libtorch_cpu.dylib)
#> frame #10: torch::autograd::backward(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, std::__1::optional<bool>, bool, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&) + 96 (0x11a03ea9c in libtorch_cpu.dylib)
#> frame #11: torch::autograd::VariableHooks::_backward(at::Tensor const&, c10::ArrayRef<at::Tensor>, std::__1::optional<at::Tensor> const&, std::__1::optional<bool>, bool) const + 220 (0x11a094c30 in libtorch_cpu.dylib)
#> frame #12: _lantern_Tensor__backward_tensor_tensorlist_tensor_bool_bool + 184 (0x1091378b4 in liblantern.dylib)
#> frame #13: std::__1::__function::__func<cpp_torch_method__backward_self_Tensor_inputs_TensorList(XPtrTorchTensor, XPtrTorchTensorList, XPtrTorchOptionalTensor, XPtrTorchoptional_bool, XPtrTorchbool)::$_1, std::__1::allocator<cpp_torch_method__backward_self_Tensor_inputs_TensorList(XPtrTorchTensor, XPtrTorchTensorList, XPtrTorchOptionalTensor, XPtrTorchoptional_bool, XPtrTorchbool)::$_1>, void ()>::operator()() + 64 (0x1078a6980 in torchpkg.so)
#> frame #14: std::__1::packaged_task<void ()>::operator()() + 80 (0x1078a4b50 in torchpkg.so)
#> frame #15: EventLoop<void>::run() + 384 (0x1078a4900 in torchpkg.so)
#> frame #16: void* std::__1::__thread_proxy[abi:ne190102]<std::__1::tuple<std::__1::unique_ptr<std::__1::__thread_struct, std::__1::default_delete<std::__1::__thread_struct>>, ThreadPool<void>::ThreadPool(int)::'lambda'()>>(void*) + 52 (0x1078a4674 in torchpkg.so)
#> frame #17: _pthread_start + 136 (0x192207bc8 in libsystem_pthread.dylib)
#> frame #18: thread_start + 8 (0x192202b80 in libsystem_pthread.dylib)
#> 
```
