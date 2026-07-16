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
#>  0.8284  0.1666  0.0892  0.3439  0.6950
#>  0.1293 -0.2954 -0.0269 -0.2101  0.6531
#>  0.1488  0.1320  0.6554  0.3681 -0.2096
#> [ CPUFloatType{3,5} ]
```
