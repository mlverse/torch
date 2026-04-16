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
#>  0.5272 -0.2015  0.3137  0.0067  0.2211
#> -0.7585 -0.0757  0.4847 -0.3076 -0.3913
#>  0.6301  0.2636  0.3723  0.4731  0.6966
#> [ CPUFloatType{3,5} ]
```
