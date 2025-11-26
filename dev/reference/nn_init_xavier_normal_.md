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
#>  0.0845  0.1385  0.1793 -1.1511  0.2486
#> -0.3120  0.3607  0.0626  0.4395  0.6315
#>  0.6212 -0.5040 -0.9026 -0.4034 -0.0221
#> [ CPUFloatType{3,5} ]
```
