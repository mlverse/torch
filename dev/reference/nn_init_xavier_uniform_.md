# Xavier uniform initialization

Fills the input `Tensor` with values according to the method described
in
`Understanding the difficulty of training deep feedforward neural networks` -
Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

## Usage

``` r
nn_init_xavier_uniform_(tensor, gain = 1)
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
nn_init_xavier_uniform_(w)
}
#> torch_tensor
#>  0.2817  0.5600  0.6032 -0.4509  0.3672
#> -0.3544 -0.6772 -0.8128 -0.2163  0.0645
#>  0.1881 -0.6288 -0.3668 -0.0501  0.7443
#> [ CPUFloatType{3,5} ]
```
