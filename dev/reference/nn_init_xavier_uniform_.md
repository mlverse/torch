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
#> -0.1809  0.3439  0.5221  0.0749 -0.6375
#> -0.0708  0.0264 -0.2484  0.3386  0.6609
#>  0.6333  0.8647 -0.1170  0.5075  0.6518
#> [ CPUFloatType{3,5} ]
```
