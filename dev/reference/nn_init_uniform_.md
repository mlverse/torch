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
#>  0.2852  0.7350  0.5566  0.5420  0.5310
#>  0.7384  0.5654  0.9115  0.0493  0.8077
#>  0.1666  0.7942  0.9854  0.1419  0.6131
#> [ CPUFloatType{3,5} ]
```
