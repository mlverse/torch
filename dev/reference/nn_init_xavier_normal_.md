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
#> -0.0912  0.0303  0.7091 -0.0537 -1.1068
#> -0.7812 -0.6150  0.8340 -0.4301  0.9662
#>  0.6114  0.1780  0.2763 -0.1446 -0.4101
#> [ CPUFloatType{3,5} ]
```
