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
#>  0.2165  0.8544  0.4610  0.3713  0.6895
#>  0.6080  0.4869  0.9411  0.1158  0.3134
#>  0.8858  0.0672  0.3254  0.3468  0.0222
#> [ CPUFloatType{3,5} ]
```
