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
#> -0.7731 -0.0159 -0.0588 -0.3188  0.5084
#>  0.1731 -0.2727  0.3934  0.3641 -0.5061
#> -1.1748  0.1267  0.9985  0.6848  0.5139
#> [ CPUFloatType{3,5} ]
```
