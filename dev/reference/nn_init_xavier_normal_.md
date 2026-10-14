# Xavier normal initialization

Fills the input `Tensor` with values according to the method described
in
`Understanding the difficulty of training deep feedforward neural networks` -
Glorot, X. & Bengio, Y. (2010), using a normal distribution.

## Usage

``` r
nn_init_xavier_normal_(tensor, gain = 1)
```

## Arguments

- tensor:

  an n-dimensional `Tensor`

- gain:

  an optional scaling factor

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_xavier_normal_(w)
}
#> torch_tensor
#>  0.9087 -0.3497 -0.3626  0.2889  0.2661
#> -1.4799 -0.1352  0.3304  0.2132 -0.2388
#>  0.3314  0.5787 -0.1715 -0.0872 -0.1596
#> [ CPUFloatType{3,5} ]
```
