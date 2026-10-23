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
#> -0.7024 -0.6849 -0.1307  0.5070 -0.3555
#>  0.7855  0.5421 -0.2482  0.5365 -0.4765
#>  0.3847  0.5123  0.6026  0.5869 -0.4753
#> [ CPUFloatType{3,5} ]
```
