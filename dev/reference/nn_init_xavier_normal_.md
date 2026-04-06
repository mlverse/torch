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
#> -0.3552  0.7203  0.2742 -0.2023 -0.3981
#> -0.1504  0.4482 -0.3131 -0.0721  0.2748
#> -0.3574 -0.3670  0.3121 -0.9664 -0.0842
#> [ CPUFloatType{3,5} ]
```
