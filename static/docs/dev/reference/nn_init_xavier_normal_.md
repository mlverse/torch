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
#> -0.7864 -0.1473  1.0706 -0.8785 -0.8935
#> -0.5024 -0.1736 -0.5432 -0.2375 -0.2501
#>  0.0286  0.0050  0.2826  0.2536 -0.4196
#> [ CPUFloatType{3,5} ]
```
