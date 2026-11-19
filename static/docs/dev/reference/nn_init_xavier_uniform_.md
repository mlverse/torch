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
#> -0.8588 -0.6975  0.1955  0.2318  0.1378
#> -0.3053  0.1403 -0.6445  0.4174 -0.1580
#>  0.3346 -0.8145 -0.4214  0.7697  0.8324
#> [ CPUFloatType{3,5} ]
```
