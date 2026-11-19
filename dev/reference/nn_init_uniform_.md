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
#>  0.7396  0.7908  0.2345  0.7576  0.5146
#>  0.3556  0.9466  0.6150  0.1446  0.5870
#>  0.8532  0.6217  0.8976  0.0406  0.7772
#> [ CPUFloatType{3,5} ]
```
