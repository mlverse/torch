# Ones initialization

Fills the input Tensor with the scalar value `1`

## Usage

``` r
nn_init_ones_(tensor)
```

## Arguments

- tensor:

  an n-dimensional `Tensor`

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_ones_(w)
}
#> torch_tensor
#>  1  1  1  1  1
#>  1  1  1  1  1
#>  1  1  1  1  1
#> [ CPUFloatType{3,5} ]
```
