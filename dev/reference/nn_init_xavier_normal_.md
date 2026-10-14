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
#> -0.3315 -0.4208  0.5450  0.1931  0.8674
#>  0.5824 -0.0277  0.1092 -0.1530  0.1676
#>  0.1026  0.2176  0.1571 -0.4297  0.4761
#> [ CPUFloatType{3,5} ]
```
