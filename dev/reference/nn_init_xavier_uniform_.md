# Xavier uniform initialization

Fills the input `Tensor` with values according to the method described
in
`Understanding the difficulty of training deep feedforward neural networks` -
Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

## Usage

``` r
nn_init_xavier_uniform_(tensor, gain = 1)
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
nn_init_xavier_uniform_(w)
}
#> torch_tensor
#> -0.2275  0.5642 -0.0709  0.1112 -0.2988
#> -0.4124  0.7330  0.6671 -0.2395  0.7140
#>  0.5268 -0.3885  0.1087  0.4646  0.3595
#> [ CPUFloatType{3,5} ]
```
