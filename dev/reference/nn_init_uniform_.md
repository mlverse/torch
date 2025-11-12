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
#>  0.4681  0.0533  0.2112  0.0867  0.3913
#>  0.2740  0.6954  0.5005  0.4993  0.7219
#>  0.4143  0.1760  0.0673  0.3254  0.1795
#> [ CPUFloatType{3,5} ]
```
