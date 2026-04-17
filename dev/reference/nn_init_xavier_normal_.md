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
#>  0.3677 -0.4340 -1.0380  0.0195  0.1180
#> -0.7859  0.2910 -0.0708  0.1287 -0.5121
#> -0.0658  0.8496 -0.6923 -0.2431 -0.0753
#> [ CPUFloatType{3,5} ]
```
