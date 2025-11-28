# TorchScript

``` r
library(torch)
```

[TorchScript](https://docs.pytorch.org/docs/stable/jit_language_reference.html#language-reference)
is a statically typed subset of Python that can be interpreted by
LibTorch without any Python dependency. The torch R package provides
interfaces to create, serialize, load and execute TorchScript programs.

Advantages of using TorchScript are:

- TorchScript code can be invoked in its own interpreter, which is
  basically a restricted Python interpreter. This interpreter does not
  acquire the Global Interpreter Lock, and so many requests can be
  processed on the same instance simultaneously.

- This format allows us to save the whole model to disk and load it into
  another environment, such as on server written in a language other
  than R.

- TorchScript gives us a representation in which we can do compiler
  optimizations on the code to make execution more efficient.

- TorchScript allows us to interface with many backend/device runtimes
  that require a broader view of the program than individual operators.

## Creating TorchScript programs

### Tracing

TorchScript programs can be created from R using tracing. When using
tracing, code is automatically converted into this subset of Python by
recording only the actual operators on tensors and simply executing and
discarding the other surrounding R code.

Currently tracing is the only supported way to create TorchScript
programs from R code.

For example, let’s use the `jit_trace` function to create a TorchScript
program. We pass a regular R function and example inputs.

``` r
fn <- function(x) {
  torch_relu(x)
}

traced_fn <- jit_trace(fn, torch_tensor(c(-1, 0, 1)))
```

The `jit_trace` function has executed the R function with the example
input and recorded all torch operations that occurred during execution
to create a *graph*. *graph* is how we call the intermediate
representation of TorchScript programs, and it can be inspected with:

``` r
traced_fn$graph
#> graph(%0 : Float(3, strides=[1], requires_grad=0, device=cpu)):
#>   %1 : Float(3, strides=[1], requires_grad=0, device=cpu) = aten::relu(%0)
#>   return (%1)
```

The traced function can now be invoked as a regular R function:

``` r
traced_fn(torch_randn(3))
#> torch_tensor
#>  0.0000
#>  0.2325
#>  1.3337
#> [ CPUFloatType{3} ]
```

It’s also possible to trace `nn_modules()` defined in R, for example:

``` r
module <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(10, 10)
    self$linear2 <- nn_linear(10, 1)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>%
      nnf_relu() %>%
      self$linear2()
  }
)
traced_module <- jit_trace(module(), torch_randn(10, 10))
```

When using `jit_trace` with a `nn_module` only the `forward` method is
traced. However, by default, one pass will be conducted in ‘train’ mode,
and another one in ‘eval’ mode, which is different from the PyTorch
behavior. One can opt out of this by specifying `respect_mode = FALSE`
which will only trace the forward pass in the mode the network is
currently in. You can use the `jit_trace_module` function to pass
example inputs to other methods. Traced modules look like normal
`nn_modules()`, and can be called the same way:

``` r
traced_module(torch_randn(3, 10))
#> torch_tensor
#> -0.2990
#> -0.9849
#> -0.4014
#> [ CPUFloatType{3,1} ][ grad_fn = <AddmmBackward0> ]
```

#### Limitations of tracing

1.  Tracing will not record any control flow like if-statements or
    loops. When this control flow is constant across your module, this
    is fine and it often inlines the control flow decisions. But
    sometimes the control flow is actually part of the model itself. For
    instance, a recurrent network is a loop over the (possibly dynamic)
    length of an input sequence. For example:

``` r
# fn does does an operation for each dimension of a tensor
fn <- function(x) {
  x %>% 
    torch_unbind(dim = 1) %>%
    lapply(function(x) x$sum()) %>%
    torch_stack(dim = 1)
}
# we trace using as an example a tensor with size (10, 5, 5)
traced_fn <- jit_trace(fn, torch_randn(10, 5, 5))
# applying it with a tensor with different size returns an error.
traced_fn(torch_randn(11, 5, 5))
#> Error in cpp_call_traced_fn(ptr, inputs): The following operation failed in the TorchScript interpreter.
#> Traceback of TorchScript (most recent call last):
#> RuntimeError: Expected 10 elements in a list but found 11
```

2.  In the returned `ScriptModule`, operations that have different
    behaviors in training and eval modes will always behave as if it
    were in the mode it was in during tracing, no matter which mode the
    `ScriptModule` is in. For example:

``` r
traced_dropout <- jit_trace(nn_dropout(), torch_ones(5,5))
traced_dropout(torch_ones(3,3))
#> torch_tensor
#>  2  0  2
#>  2  0  0
#>  0  2  0
#> [ CPUFloatType{3,3} ]
traced_dropout$eval()
#> [1] FALSE
# even after setting to eval mode, dropout is applied
traced_dropout(torch_ones(3,3))
#> torch_tensor
#>  1  1  1
#>  1  1  1
#>  1  1  1
#> [ CPUFloatType{3,3} ]
```

3.  Tracing proegrams can only take tensors and lists of tensors as
    input and return tensors and lists of tensors. For example:

``` r
fn <- function(x, y) {
  x + y
}
jit_trace(fn, torch_tensor(1), 1)
#> Error in cpp_trace_function(tr_fn, list(...), .compilation_unit, strict, : Only tensors or (possibly nested) dict or tuples of tensors can be inputs to traced functions. Got float
#> Exception raised from addInput at /Users/runner/work/libtorch-mac-m1/libtorch-mac-m1/pytorch/torch/csrc/jit/frontend/tracer.cpp:424 (most recent call first):
#> frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 52 (0x103e1c55c in libc10.dylib)
#> frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) + 140 (0x103e191ac in libc10.dylib)
#> frame #2: torch::jit::tracer::addInput(std::__1::shared_ptr<torch::jit::tracer::TracingState> const&, c10::IValue const&, c10::Type::SingletonOrSharedTypePtr<c10::Type> const&, torch::jit::Value*) + 5648 (0x147fdfda4 in libtorch_cpu.dylib)
#> frame #3: torch::jit::tracer::addInput(std::__1::shared_ptr<torch::jit::tracer::TracingState> const&, c10::IValue const&, c10::Type::SingletonOrSharedTypePtr<c10::Type> const&, torch::jit::Value*) + 4268 (0x147fdf840 in libtorch_cpu.dylib)
#> frame #4: torch::jit::tracer::trace(std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue>>, std::__1::function<std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue>> (std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue>>)> const&, std::__1::function<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> (at::Tensor const&)>, bool, bool, torch::jit::Module*, std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>> const&) + 680 (0x147fdd26c in libtorch_cpu.dylib)
#> frame #5: _lantern_trace_fn + 292 (0x12b0c5dd4 in liblantern.dylib)
#> frame #6: cpp_trace_function(Rcpp::Function_Impl<Rcpp::PreserveStorage>, XPtrTorchStack, XPtrTorchCompilationUnit, XPtrTorchstring, bool, XPtrTorchScriptModule, bool, bool) + 524 (0x10b48b78c in torchpkg.so)
#> frame #7: _torch_cpp_trace_function + 688 (0x10b27d3f0 in torchpkg.so)
#> frame #8: R_doDotCall + 3356 (0x1012100dc in libR.dylib)
#> frame #9: bcEval_loop + 81268 (0x10125fbf4 in libR.dylib)
#> frame #10: bcEval + 592 (0x10123e3d0 in libR.dylib)
#> frame #11: Rf_eval + 556 (0x10123db6c in libR.dylib)
#> frame #12: R_execClosure + 816 (0x101240730 in libR.dylib)
#> frame #13: applyClosure_core + 164 (0x10123f824 in libR.dylib)
#> frame #14: Rf_eval + 1224 (0x10123de08 in libR.dylib)
#> frame #15: do_eval + 1352 (0x101244ec8 in libR.dylib)
#> frame #16: bcEval_loop + 27244 (0x1012528ec in libR.dylib)
#> frame #17: bcEval + 592 (0x10123e3d0 in libR.dylib)
#> frame #18: Rf_eval + 556 (0x10123db6c in libR.dylib)
#> frame #19: forcePromise + 232 (0x10123e668 in libR.dylib)
#> frame #20: Rf_eval + 660 (0x10123dbd4 in libR.dylib)
#> frame #21: do_withVisible + 64 (0x101245200 in libR.dylib)
#> frame #22: do_internal + 400 (0x1012a4490 in libR.dylib)
#> frame #23: bcEval_loop + 27828 (0x101252b34 in libR.dylib)
#> frame #24: bcEval + 592 (0x10123e3d0 in libR.dylib)
#> frame #25: Rf_eval + 556 (0x10123db6c in libR.dylib)
#> frame #26: forcePromise + 232 (0x10123e668 in libR.dylib)
#> frame #27: Rf_eval + 660 (0x10123dbd4 in libR.dylib)
#> frame #28: forcePromise + 232 (0x10123e668 in libR.dylib)
#> frame #29: getvar + 408 (0x101261a18 in libR.dylib)
#> frame #30: bcEval_loop + 16936 (0x1012500a8 in libR.dylib)
#> frame #31: bcEval + 592 (0x10123e3d0 in libR.dylib)
#> frame #32: Rf_eval + 556 (0x10123db6c in libR.dylib)
#> frame #33: R_execClosure + 816 (0x101240730 in libR.dylib)
#> frame #34: applyClosure_core + 164 (0x10123f824 in libR.dylib)
#> frame #35: Rf_eval + 1224 (0x10123de08 in libR.dylib)
#> frame #36: do_eval + 1352 (0x101244ec8 in libR.dylib)
#> frame #37: bcEval_loop + 27244 (0x1012528ec in libR.dylib)
#> frame #38: bcEval + 592 (0x10123e3d0 in libR.dylib)
#> frame #39: Rf_eval + 556 (0x10123db6c in libR.dylib)
#> frame #40: R_execClosure + 816 (0x101240730 in libR.dylib)
#> frame #41: applyClosure_core + 164 (0x10123f824 in libR.dylib)
#> frame #42: Rf_eval + 1224 (0x10123de08 in libR.dylib)
#> frame #43: do_begin + 400 (0x101242fd0 in libR.dylib)
#> frame #44: Rf_eval + 1012 (0x10123dd34 in libR.dylib)
#> frame #45: R_execClosure + 816 (0x101240730 in libR.dylib)
#> frame #46: applyClosure_core + 164 (0x10123f824 in libR.dylib)
#> frame #47: Rf_eval + 1224 (0x10123de08 in libR.dylib)
#> frame #48: do_docall + 628 (0x1011dc3b4 in libR.dylib)
#> frame #49: bcEval_loop + 27244 (0x1012528ec in libR.dylib)
#> frame #50: bcEval + 592 (0x10123e3d0 in libR.dylib)
#> frame #51: Rf_eval + 556 (0x10123db6c in libR.dylib)
#> frame #52: R_execClosure + 816 (0x101240730 in libR.dylib)
#> frame #53: applyClosure_core + 164 (0x10123f824 in libR.dylib)
#> frame #54: Rf_eval + 1224 (0x10123de08 in libR.dylib)
#> frame #55: do_docall + 628 (0x1011dc3b4 in libR.dylib)
#> frame #56: bcEval_loop + 27244 (0x1012528ec in libR.dylib)
#> frame #57: bcEval + 592 (0x10123e3d0 in libR.dylib)
#> frame #58: Rf_eval + 556 (0x10123db6c in libR.dylib)
#> frame #59: R_execClosure + 816 (0x101240730 in libR.dylib)
#> frame #60: applyClosure_core + 164 (0x10123f824 in libR.dylib)
#> frame #61: Rf_eval + 1224 (0x10123de08 in libR.dylib)
#> frame #62: forcePromise + 232 (0x10123e668 in libR.dylib)
#> :
```

### Compiling TorchScript

It’s also possible to create TorchScript programs by compiling
TorchScript code. TorchScript code looks a lot like standard python
code. For example:

``` r
tr <- jit_compile("
def fn (x: Tensor):
  return torch.relu(x)

")
tr$fn(torch_tensor(c(-1, 0, 1)))
#> torch_tensor
#>  0
#>  0
#>  1
#> [ CPUFloatType{3} ]
```

## Serializing and loading

TorchScript programs can be serialized using the `jit_save` function and
loaded back from disk with `jit_load`.

For example:

``` r
fn <- function(x) {
  torch_relu(x)
}
tr_fn <- jit_trace(fn, torch_tensor(1))
jit_save(tr_fn, "path.pt")
loaded <- jit_load("path.pt")
```

Loaded programs can be executed as usual:

``` r
loaded(torch_tensor(c(-1, 0, 1)))
#> torch_tensor
#>  0
#>  0
#>  1
#> [ CPUFloatType{3} ]
```

**Note** You can load TorchScript programs that were created in
libraries different than `torch` for R. Eg, a TorchScript program can be
created in PyTorch with `torch.jit.trace` or `torch.jit.script`, and run
from R.

R objects are automatically converted to their TorchScript counterpart
following the Types table in this document. However, sometimes it’s
necessary to make type annotations with
[`jit_tuple()`](https://torch.mlverse.org/docs/dev/reference/jit_tuple.md)
and
[`jit_scalar()`](https://torch.mlverse.org/docs/dev/reference/jit_scalar.md)
to disambiguate the conversion.

## Types

The following table lists all TorchScript types and how to convert the
to and back to R.

| TorchScript Type          | R Description                                                                                                                                                                 |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Tensor`                  | A `torch_tensor` with any shape, dtype or backend.                                                                                                                            |
| `Tuple[T0, T1, ..., TN]`  | A [`list()`](https://rdrr.io/r/base/list.html) containing subtypes `T0`, `T1`, etc. wrapped with [`jit_tuple()`](https://torch.mlverse.org/docs/dev/reference/jit_tuple.md) . |
| `bool`                    | A scalar logical value create using `jit_scalar`.                                                                                                                             |
| `int`                     | A scalar integer value created using `jit_scalar`.                                                                                                                            |
| `float`                   | A scalar floating value created using `jit_scalar`.                                                                                                                           |
| `str`                     | A string (ie. character vector of length 1) wrapped in `jit_scalar`.                                                                                                          |
| `List[T]`                 | An R list of which all types are type `T` . Or numeric vectors, logical vectors, etc.                                                                                         |
| `Optional[T]`             | Not yet supported.                                                                                                                                                            |
| `Dict[str, V]`            | A named list with values of type `V` . Only `str` key values are currently supported.                                                                                         |
| `T`                       | Not yet supported.                                                                                                                                                            |
| `E`                       | Not yet supported.                                                                                                                                                            |
| `NamedTuple[T0, T1, ...]` | A named list containing subtypes `T0`, `T1`, etc. wrapped in [`jit_tuple()`](https://torch.mlverse.org/docs/dev/reference/jit_tuple.md).                                      |
