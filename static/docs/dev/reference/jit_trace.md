# Trace a function and return an executable `script_function`.

Using `jit_trace`, you can turn an existing R function into a
TorchScript `script_function`. You must provide example inputs, and we
run the function, recording the operations performed on all the tensors.

## Usage

``` r
jit_trace(func, ..., strict = TRUE, respect_mode = TRUE)
```

## Arguments

- func:

  An R function that will be run with `example_inputs`. func arguments
  and return values must be tensors or (possibly nested) lists that
  contain tensors. Can also be a
  [`nn_module()`](https://torch.mlverse.org/docs/dev/reference/nn_module.md),
  in such case
  [`jit_trace_module()`](https://torch.mlverse.org/docs/dev/reference/jit_trace_module.md)
  is used to trace that module.

- ...:

  example inputs that will be passed to the function while tracing. The
  resulting trace can be run with inputs of different types and shapes
  assuming the traced operations support those types and shapes.
  `example_inputs` may also be a single Tensor in which case it is
  automatically wrapped in a list. Note that `...` **can not** be named,
  and the order is respected.

- strict:

  run the tracer in a strict mode or not (default: `TRUE`). Only turn
  this off when you want the tracer to record your mutable container
  types (currently list/dict) and you are sure that the container you
  are using in your problem is a constant structure and does not get
  used as control flow (`if`, `for`) conditions.

- respect_mode:

  (`logical(1)`)  
  Whether both modes ('train' or 'eval') should be traced. If `TRUE`
  (default), the underlying C++ ScriptModule will have two methods
  `trainforward()` and `evalforward()`. The `$forward()` method of the R
  torch module will then select either based on the mode. If `FALSE`,
  only the current mode of the module will be jitted and hence only one
  `forward()` method exists.

## Value

An `script_function` if `func` is a function and `script_module` if
`func` is a
[`nn_module()`](https://torch.mlverse.org/docs/dev/reference/nn_module.md).

## Details

The resulting recording of a standalone function produces a
`script_function`.

## Warning

Tracing only correctly records functions and modules which are not data
dependent (e.g., do not have conditionals on data in tensors) and do not
have any untracked external dependencies (e.g., perform input/output or
access global variables). Tracing only records operations done when the
given function is run on the given tensors. Therefore, the returned
`script_function` will always run the same traced graph on any input.
This has some important implications when your module is expected to run
different sets of operations, depending on the input and/or the module
state. For example,

- Tracing will not record any control-flow like if-statements or loops.
  When this control-flow is constant across your module, this is fine
  and it often inlines the control-flow decisions. But sometimes the
  control-flow is actually part of the model itself. For instance, a
  recurrent network is a loop over the (possibly dynamic) length of an
  input sequence.

- In the returned `script_function`, operations that have different
  behaviors in training and eval modes will always behave as if it is in
  the mode it was in during tracing, no matter which mode the
  `script_function` is in.

In cases like these, tracing would not be appropriate and scripting is a
better choice. If you trace such models, you may silently get incorrect
results on subsequent invocations of the model. The tracer will try to
emit warnings when doing something that may cause an incorrect trace to
be produced. For scripting, see
[`jit_compile`](https://torch.mlverse.org/docs/dev/reference/jit_compile.md).

## Examples

``` r
if (torch_is_installed()) {
fn <- function(x) {
  torch_relu(x)
}
input <- torch_tensor(c(-1, 0, 1))
tr_fn <- jit_trace(fn, input)
tr_fn(input)
}
#> torch_tensor
#>  0
#>  0
#>  1
#> [ CPUFloatType{3} ]
```
