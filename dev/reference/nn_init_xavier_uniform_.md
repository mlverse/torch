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
#> -0.4233 -0.2505 -0.1556  0.6625  0.4267
#> -0.6906  0.0030  0.1657  0.2036 -0.5895
#> -0.5218 -0.2326 -0.0811 -0.1683  0.7245
#> [ CPUFloatType{3,5} ]
```
