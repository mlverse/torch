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
#>  0.1851  0.2329  0.2582  0.6639  0.1759
#>  0.2929  0.5301  0.2181  0.0947  0.2519
#>  0.9577  0.1072  0.9600  0.1391  0.0190
#> [ CPUFloatType{3,5} ]
```
