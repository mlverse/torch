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
#>  0.4165  0.1655  0.1298  0.6015  0.9181
#>  0.5132  0.1651  0.5203  0.0389  0.4959
#>  0.9166  0.8072  0.1144  0.9537  0.6065
#> [ CPUFloatType{3,5} ]
```
