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
#>  0.2266 -0.2696 -0.2341 -0.0850 -1.0080
#>  0.1951 -0.2912 -0.0336  0.8923  0.0413
#> -0.8290  0.6355 -0.6854  0.0675 -0.8599
#> [ CPUFloatType{3,5} ]
```
