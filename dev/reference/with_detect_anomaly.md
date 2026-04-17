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
#> Error : one of the variables needed for gradient computation has been modified by an inplace operation: [CPUFloatType [1]] is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
#> Exception raised from unpack at /Users/runner/work/libtorch-mac-m1/libtorch-mac-m1/pytorch/torch/csrc/autograd/saved_variable.cpp:194 (most recent call first):
#> frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 56 (0x107f7ab74 in libc10.dylib)
#> frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) + 120 (0x107f780e8 in libc10.dylib)
#> frame #2: torch::autograd::SavedVariable::unpack(std::__1::shared_ptr<torch::autograd::Node>) const + 2300 (0x14c7c7644 in libtorch_cpu.dylib)
#> frame #3: torch::autograd::generated::PowBackward1::apply(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>>&&) + 60 (0x14b70a3b4 in libtorch_cpu.dylib)
#> frame #4: torch::autograd::Node::operator()(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>>&&) + 128 (0x14c794fdc in libtorch_cpu.dylib)
#> frame #5: torch::autograd::Engine::evaluate_function(std::__1::shared_ptr<torch::autograd::GraphTask>&, torch::autograd::Node*, torch::autograd::InputBuffer&, std::__1::shared_ptr<torch::autograd::ReadyQueue> const&) + 3196 (0x14c78d1f0 in libtorch_cpu.dylib)
#> frame #6: torch::autograd::Engine::thread_main(std::__1::shared_ptr<torch::autograd::GraphTask> const&) + 844 (0x14c78be8c in libtorch_cpu.dylib)
#> frame #7: torch::autograd::Engine::execute_with_graph_task(std::__1::shared_ptr<torch::autograd::GraphTask> const&, std::__1::shared_ptr<torch::autograd::Node>, torch::autograd::InputBuffer&&) + 660 (0x14c793b54 in libtorch_cpu.dylib)
#> frame #8: torch::autograd::Engine::execute(std::__1::vector<torch::autograd::Edge, std::__1::allocator<torch::autograd::Edge>> const&, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, bool, bool, bool, std::__1::vector<torch::autograd::Edge, std::__1::allocator<torch::autograd::Edge>> const&) + 2088 (0x14c79299c in libtorch_cpu.dylib)
#> frame #9: torch::autograd::run_backward(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, bool, bool, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, bool, bool) + 832 (0x14c77a694 in libtorch_cpu.dylib)
#> frame #10: torch::autograd::backward(std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&, std::__1::optional<bool>, bool, std::__1::vector<at::Tensor, std::__1::allocator<at::Tensor>> const&) + 88 (0x14c779bbc in libtorch_cpu.dylib)
#> frame #11: torch::autograd::VariableHooks::_backward(at::Tensor const&, c10::ArrayRef<at::Tensor>, std::__1::optional<at::Tensor> const&, std::__1::optional<bool>, bool) const + 412 (0x14c7cc4f4 in libtorch_cpu.dylib)
#> frame #12: _lantern_Tensor__backward_tensor_tensorlist_tensor_bool_bool + 184 (0x10a765c34 in liblantern.dylib)
#> frame #13: std::__1::__function::__func<cpp_torch_method__backward_self_Tensor_inputs_TensorList(XPtrTorchTensor, XPtrTorchTensorList, XPtrTorchOptionalTensor, XPtrTorchoptional_bool, XPtrTorchbool)::$_1, std::__1::allocator<cpp_torch_method__backward_self_Tensor_inputs_TensorList(XPtrTorchTensor, XPtrTorchTensorList, XPtrTorchOptionalTensor, XPtrTorchoptional_bool, XPtrTorchbool)::$_1>, void ()>::operator()() + 64 (0x109cc4b80 in torchpkg.so)
#> frame #14: std::__1::packaged_task<void ()>::operator()() + 80 (0x109cc2d50 in torchpkg.so)
#> frame #15: EventLoop<void>::run() + 384 (0x109cc2b00 in torchpkg.so)
#> frame #16: void* std::__1::__thread_proxy[abi:ne190102]<std::__1::tuple<std::__1::unique_ptr<std::__1::__thread_struct, std::__1::default_delete<std::__1::__thread_struct>>, ThreadPool<void>::ThreadPool(int)::'lambda'()>>(void*) + 52 (0x109cc2874 in torchpkg.so)
#> frame #17: _pthread_start + 136 (0x19f61bbc8 in libsystem_pthread.dylib)
#> frame #18: thread_start + 8 (0x19f616b80 in libsystem_pthread.dylib)
#> 
```
