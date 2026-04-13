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
#> -0.3424 -1.0268 -0.8959  0.7186 -0.3487
#> -0.1496 -0.2553  0.4693 -0.7913  0.6224
#> -0.3163  0.6416 -1.0134 -0.0789 -0.8193
#> [ CPUFloatType{3,5} ]
```
