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
#>  0.7741  0.2937  0.4539  0.0707  0.8532
#>  0.2167  0.0434  0.5048  0.8583  0.5458
#>  0.4961  0.5300  0.4700  0.6921  0.4267
#> [ CPUFloatType{3,5} ]
```
