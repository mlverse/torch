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
#>  0.9009  0.8973  0.8308  0.5179  0.5970
#>  0.9103  0.7178  0.1803  0.3706  0.3821
#>  0.8055  0.6069  0.7348  0.7846  0.5440
#> [ CPUFloatType{3,5} ]
```
