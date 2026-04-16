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
#> -0.6889 -0.8326 -0.1753  0.1365  0.0708
#> -0.5056 -0.6960 -0.5343  0.2861  0.5112
#> -0.2583  0.7215  0.5215  0.1346  0.0863
#> [ CPUFloatType{3,5} ]
```
