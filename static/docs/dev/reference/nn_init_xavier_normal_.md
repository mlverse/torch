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
#> -0.1813  0.0071  0.3378  1.7652 -0.1040
#> -0.3293  0.3899  0.1770 -0.5063  0.0970
#>  0.1489  0.3691  0.7336  0.3501 -0.0885
#> [ CPUFloatType{3,5} ]
```
