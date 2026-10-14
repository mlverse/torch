# Enable idiomatic access to JIT operators from R.

Call JIT operators directly from R, keeping the familiar argument types
and argument order. Note, however, that:

- all arguments are required (no defaults)

- axis numbering (as well as position numbers overall) starts from 0

- scalars have to be wrapped in
  [`jit_scalar()`](https://torch.mlverse.org/docs/dev/reference/jit_scalar.md)

## Usage

``` r
jit_ops
```

## Format

An object of class `torch_ops` of length 0.

## Examples

``` r
if (torch_is_installed()) {
t1 <- torch::torch_rand(4, 5)
t2 <- torch::torch_ones(5, 4)
# same as torch::torch_matmul(t1, t2)
jit_ops$aten$matmul(t1, t2)

# same as torch_split(torch::torch_arange(0, 3), 2, 1)
jit_ops$aten$split(torch::torch_arange(0, 3), torch::jit_scalar(2L), torch::jit_scalar(0L))

}
#> [[1]]
#> torch_tensor
#>  0
#>  1
#> [ CPUFloatType{2} ]
#> 
#> [[2]]
#> torch_tensor
#>  2
#>  3
#> [ CPUFloatType{2} ]
#> 
```
