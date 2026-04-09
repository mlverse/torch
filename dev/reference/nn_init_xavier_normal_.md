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
#>  0.5851  0.0476  0.1637  0.8168  0.0511
#> -0.4953  0.4563 -0.3729  0.9526  0.2137
#>  0.9258  0.2108 -0.0662  0.3951 -0.3715
#> [ CPUFloatType{3,5} ]
```
