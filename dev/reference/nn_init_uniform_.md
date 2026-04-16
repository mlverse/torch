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
#>  0.0676  0.7111  0.8540  0.0168  0.6137
#>  0.2284  0.8711  0.9806  0.3750  0.6765
#>  0.0421  0.1984  0.7156  0.9750  0.7173
#> [ CPUFloatType{3,5} ]
```
