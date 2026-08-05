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
#>  0.9831  0.0255  0.7558  0.7369  0.5581
#>  0.6558  0.7845  0.4404  0.6598  0.5428
#>  0.5078  0.6640  0.2546  0.4482  0.9719
#> [ CPUFloatType{3,5} ]
```
