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
#> -0.7417 -0.7268  0.6929  0.1130  0.5770
#> -0.4083  1.0739  0.0397  0.7003 -0.0719
#>  0.0814  0.2997 -0.0532 -0.2381 -0.3068
#> [ CPUFloatType{3,5} ]
```
