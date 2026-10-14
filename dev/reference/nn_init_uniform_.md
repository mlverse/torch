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
#>  0.1121  0.9147  0.3697  0.4613  0.7390
#>  0.7225  0.8027  0.9274  0.1899  0.0492
#>  0.6078  0.1457  0.7986  0.1435  0.4620
#> [ CPUFloatType{3,5} ]
```
