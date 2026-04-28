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
#>  0.0372  0.5287  0.9839 -0.8681 -0.3594
#>  0.0413  0.2698  1.0696  0.1296 -0.8632
#> -0.0324  0.5747 -0.5035  0.1965  0.5079
#> [ CPUFloatType{3,5} ]
```
