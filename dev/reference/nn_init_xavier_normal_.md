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
#>  0.1637  0.4710  0.2131  0.1951 -0.9730
#> -0.3558  0.4787 -0.1350 -0.1619 -1.3306
#> -0.3008  0.2346 -0.2043 -0.6582  0.3207
#> [ CPUFloatType{3,5} ]
```
