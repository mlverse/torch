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
#>  0.4067  0.1186  0.7614  0.2105  0.9714
#>  0.4584  0.0730  0.4204  0.2295  0.4079
#>  0.9898  0.0464  0.5631  0.4159  0.6251
#> [ CPUFloatType{3,5} ]
```
