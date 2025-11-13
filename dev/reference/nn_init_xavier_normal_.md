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
#> -0.3743 -0.0215 -0.0991 -1.0206  0.4338
#>  0.0083 -0.2269 -0.2734  0.2420 -0.1330
#> -0.5260 -0.5826 -0.1225  0.6735 -0.3648
#> [ CPUFloatType{3,5} ]
```
