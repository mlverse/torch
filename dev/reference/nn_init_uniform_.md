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
#>  0.0549  0.4697  0.8058  0.9872  0.8151
#>  0.4708  0.7650  0.1018  0.7263  0.3710
#>  0.5840  0.9974  0.2363  0.6067  0.2953
#> [ CPUFloatType{3,5} ]
```
