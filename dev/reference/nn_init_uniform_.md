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
#>  0.4374  0.6673  0.1194  0.5905  0.3129
#>  0.7325  0.1531  0.4070  0.6240  0.7619
#>  0.6574  0.9437  0.3100  0.5421  0.2559
#> [ CPUFloatType{3,5} ]
```
