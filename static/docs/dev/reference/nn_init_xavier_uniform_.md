# Xavier uniform initialization

Fills the input `Tensor` with values according to the method described
in
`Understanding the difficulty of training deep feedforward neural networks` -
Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

## Usage

``` r
nn_init_xavier_uniform_(tensor, gain = 1)
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
nn_init_xavier_uniform_(w)
}
#> torch_tensor
#> -0.3957  0.1153 -0.7236  0.2278  0.6969
#> -0.6408  0.1596 -0.0071  0.0615  0.6270
#>  0.1599 -0.6373 -0.5541  0.1261 -0.6967
#> [ CPUFloatType{3,5} ]
```
