# Compile TorchScript code into a graph

See the [TorchScript language
reference](https://docs.pytorch.org/docs/stable/jit_language_reference.html#language-reference)
for documentation on how to write TorchScript code.

## Usage

``` r
jit_compile(source)
```

## Arguments

- source:

  valid TorchScript source code.

## Examples

``` r
if (torch_is_installed()) {
comp <- jit_compile("
def fn (x):
  return torch.abs(x)

def foo (x):
  return torch.sum(x)

")

comp$fn(torch_tensor(-1))
comp$foo(torch_randn(10))
}
#> torch_tensor
#> 1.63059
#> [ CPUFloatType{} ]
```
