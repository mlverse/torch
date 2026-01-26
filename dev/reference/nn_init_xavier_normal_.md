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
#> -0.1012 -1.5029  0.4287 -0.2844 -0.3367
#> -0.3648 -0.0163  1.1222  0.4944  0.3324
#> -0.4661 -0.3930 -0.4349 -0.5508 -0.4497
#> [ CPUFloatType{3,5} ]
```
