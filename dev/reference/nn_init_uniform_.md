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
#>  0.0674  0.4517  0.6586  0.0069  0.3441
#>  0.5048  0.2745  0.6873  0.8276  0.7432
#>  0.1070  0.3045  0.4388  0.4683  0.5017
#> [ CPUFloatType{3,5} ]
```
