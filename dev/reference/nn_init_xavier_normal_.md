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
#> -0.4944  0.0810  0.1031  0.2957  0.3469
#>  0.4696 -0.2339  0.7691 -0.1608 -0.2868
#> -0.1088  0.5009  0.3005  0.3019  0.0040
#> [ CPUFloatType{3,5} ]
```
