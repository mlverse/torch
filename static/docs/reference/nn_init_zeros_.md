# Zeros initialization

Fills the input Tensor with the scalar value `0`

## Usage

``` r
nn_init_zeros_(tensor)
```

## Arguments

- tensor:

  an n-dimensional tensor

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_zeros_(w)
}
#> torch_tensor
#>  0  0  0  0  0
#>  0  0  0  0  0
#>  0  0  0  0  0
#> [ CPUFloatType{3,5} ]
```
