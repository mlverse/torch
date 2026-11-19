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
#>  0.2242 -1.1448  0.4269  0.3308 -0.4103
#> -0.2068 -0.5633 -0.2517  0.5765  0.2002
#> -0.6237 -0.5038 -0.0536 -0.6221 -0.2528
#> [ CPUFloatType{3,5} ]
```
