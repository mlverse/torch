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
#> -0.5985  0.8727 -0.0260 -0.5396  0.0356
#>  0.1361 -0.2033 -0.7995  0.6151  0.1586
#>  0.5449 -0.1724  0.4651 -0.3702  0.3178
#> [ CPUFloatType{3,5} ]
```
