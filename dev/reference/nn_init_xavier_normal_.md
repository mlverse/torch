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
#>  0.1065  0.4548  1.2283  0.2977 -0.0144
#> -0.2799  0.4253 -1.0470  0.3512  0.5591
#>  0.6109  0.3144 -0.8120 -0.0758  0.3426
#> [ CPUFloatType{3,5} ]
```
