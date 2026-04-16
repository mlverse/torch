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
#>  0.7219  0.4403 -0.4097  1.3407  0.8593
#>  0.4467 -0.0652  0.6156  0.0112  0.3495
#> -0.3227  0.0231 -0.4805  0.1848 -0.7330
#> [ CPUFloatType{3,5} ]
```
