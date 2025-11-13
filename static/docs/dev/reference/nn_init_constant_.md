# Constant initialization

Fills the input Tensor with the value `val`.

## Usage

``` r
nn_init_constant_(tensor, val)
```

## Arguments

- tensor:

  an n-dimensional `Tensor`

- val:

  the value to fill the tensor with

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_constant_(w, 0.3)
}
#> torch_tensor
#>  0.3000  0.3000  0.3000  0.3000  0.3000
#>  0.3000  0.3000  0.3000  0.3000  0.3000
#>  0.3000  0.3000  0.3000  0.3000  0.3000
#> [ CPUFloatType{3,5} ]
```
