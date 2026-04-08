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
#>  0.1578  0.8730  0.7527  0.8476  0.0838
#>  0.3975  0.5167  0.4524  0.4183  0.4299
#>  0.5929  0.9264  0.2900  0.8098  0.7877
#> [ CPUFloatType{3,5} ]
```
