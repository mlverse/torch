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
#> -0.8332 -0.0305 -1.1676 -0.2149  0.0132
#>  0.3757  0.1553  0.2824 -0.7227 -0.1238
#> -0.3270 -0.7129  0.2276  0.3329 -0.4274
#> [ CPUFloatType{3,5} ]
```
