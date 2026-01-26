# Uniform initialization

Fills the input Tensor with values drawn from the uniform distribution

## Usage

``` r
nn_init_uniform_(tensor, a = 0, b = 1)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- a:

  the lower bound of the uniform distribution

- b:

  the upper bound of the uniform distribution

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_uniform_(w)
}
#> torch_tensor
#>  0.2538  0.7110  0.0015  0.0545  0.2539
#>  0.5462  0.7385  0.7366  0.9355  0.8055
#>  0.2042  0.6747  0.6035  0.4467  0.7010
#> [ CPUFloatType{3,5} ]
```
