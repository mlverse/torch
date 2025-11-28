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
#>  0.0111 -0.7521 -0.2166 -1.4364 -0.1252
#>  0.5491  0.6864  0.6320 -0.1544 -0.4447
#> -0.0655  0.2040  0.3490  0.1356  0.3134
#> [ CPUFloatType{3,5} ]
```
