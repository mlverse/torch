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
#>  0.0386  0.8494  0.3299  0.7353  0.2553
#>  0.1556  0.7051  0.8159  0.7411  0.9682
#>  0.6191  0.0756  0.8081  0.8130  0.0972
#> [ CPUFloatType{3,5} ]
```
