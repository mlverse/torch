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
#>  0.7662  0.9353  0.4825  0.7332  0.9775
#>  0.2365  0.4832  0.9215  0.8727  0.3120
#>  0.0245  0.6850  0.5385  0.7156  0.4574
#> [ CPUFloatType{3,5} ]
```
