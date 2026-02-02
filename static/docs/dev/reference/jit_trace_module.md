# Trace a module

Trace a module and return an executable ScriptModule that will be
optimized using just-in-time compilation. When a module is passed to
[`jit_trace()`](https://torch.mlverse.org/docs/dev/reference/jit_trace.md),
only the forward method is run and traced. With `jit_trace_module()`,
you can specify a named list of method names to example inputs to trace
(see the inputs) argument below.

## Usage

``` r
jit_trace_module(mod, ..., strict = TRUE, respect_mode = TRUE)
```

## Arguments

- mod:

  A torch
  [`nn_module()`](https://torch.mlverse.org/docs/dev/reference/nn_module.md)
  containing methods whose names are specified in inputs. The given
  methods will be compiled as a part of a single ScriptModule.

- ...:

  A named list containing sample inputs indexed by method names in mod.
  The inputs will be passed to methods whose names correspond to inputs
  keys while tracing.
  `list('forward'=example_forward_input, 'method2'=example_method2_input)`.

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

## Details

See
[jit_trace](https://torch.mlverse.org/docs/dev/reference/jit_trace.md)
for more information on tracing.

## Examples

``` r
if (torch_is_installed()) {
linear <- nn_linear(10, 1)
tr_linear <- jit_trace_module(linear, forward = list(torch_randn(10, 10)))

x <- torch_randn(10, 10)
torch_allclose(linear(x), tr_linear(x))
}
#> [1] TRUE
```
