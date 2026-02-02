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
#> -0.1753 -0.3503  0.4050 -0.2354 -0.1520
#>  0.1078 -0.1264  0.9863  0.0890  0.1198
#>  0.2183  0.5032  0.4506  0.5602  0.4413
#> [ CPUFloatType{3,5} ]
```
