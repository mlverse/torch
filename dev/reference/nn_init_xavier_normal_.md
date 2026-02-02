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
#>  0.1571  0.1080  0.1504 -0.4818  0.3056
#> -0.1733 -0.9466 -0.5009 -0.1697  0.2231
#> -0.3127  0.2275  0.1703  0.3099 -1.0092
#> [ CPUFloatType{3,5} ]
```
