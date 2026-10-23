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
#> -0.2162  0.1570  0.6597 -0.5018  0.5267
#> -0.6369 -0.3498  0.2496  0.5057  0.0659
#>  0.8558  0.4896  0.6660  0.8431 -0.0797
#> [ CPUFloatType{3,5} ]
```
