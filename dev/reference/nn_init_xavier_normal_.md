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
#> -0.5309 -0.0363  0.8475 -0.2906  0.6190
#> -0.8237 -0.3462 -0.8911  0.4880 -0.7507
#>  0.4141  0.3447  0.0018 -1.0272 -0.7390
#> [ CPUFloatType{3,5} ]
```
