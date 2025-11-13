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
#>  0.6476  0.9366  0.0247  0.6823  0.9558
#>  0.1813  0.8537  0.6799  0.5659  0.8661
#>  0.3741  0.0273  0.9147  0.7461  0.1277
#> [ CPUFloatType{3,5} ]
```
