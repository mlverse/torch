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
#>  0.4828  0.6564  0.4070  0.0823  0.5306
#>  0.1756  0.9536  0.8806  0.9866  0.6682
#>  0.7388  0.6514  0.6142  0.6842  0.4457
#> [ CPUFloatType{3,5} ]
```
