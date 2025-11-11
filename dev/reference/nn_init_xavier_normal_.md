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
#>  0.1720  0.6462 -0.1399  0.8018  0.5419
#>  0.3257  0.0700 -0.0630 -0.6813  0.8160
#> -0.0795 -0.4460  0.2377 -0.1596 -0.1561
#> [ CPUFloatType{3,5} ]
```
