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
#>  0.9899  0.4769 -0.4782  0.3896  0.4869
#>  0.2996  0.1201  0.0011 -0.9352 -0.0121
#> -0.0627  0.3770 -0.2755 -0.5258 -0.5476
#> [ CPUFloatType{3,5} ]
```
