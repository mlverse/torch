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
#> -0.0262  0.1330 -0.2026  0.2143  0.2015
#> -0.1224  0.4251  0.4711  0.0539 -0.3040
#> -0.1884 -0.4407 -1.1389  0.8958  0.5224
#> [ CPUFloatType{3,5} ]
```
