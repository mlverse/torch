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
#>  0.4393  0.5939  0.8662  0.4534  0.5273
#>  0.9855  0.5021  0.5785  0.1974  0.5535
#>  0.5641  0.9832  0.1070  0.7396  0.7409
#> [ CPUFloatType{3,5} ]
```
