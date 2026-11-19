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
#>  0.3155 -0.1867 -0.1711  0.5561  0.9266
#>  0.7939 -0.3322 -0.0245  0.4206 -0.0569
#> -0.4777  0.7389  0.4779 -0.3008 -0.2857
#> [ CPUFloatType{3,5} ]
```
