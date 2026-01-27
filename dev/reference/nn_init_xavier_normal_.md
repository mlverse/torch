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
#>  0.2991  0.7635 -0.5971 -0.3183 -0.1538
#> -0.0744  0.4314 -0.8764 -0.1628 -0.5221
#>  1.0675  0.2082 -0.7368  0.4182 -0.0904
#> [ CPUFloatType{3,5} ]
```
