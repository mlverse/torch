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
#> -0.3941  0.0786  0.1573  0.2639  0.5458
#> -0.1085  0.6177  0.1450 -0.3599  0.3651
#> -0.3338 -0.5265 -0.6901  0.4204 -0.8724
#> [ CPUFloatType{3,5} ]
```
