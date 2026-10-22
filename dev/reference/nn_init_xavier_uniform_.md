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
#> -0.0046  0.5631 -0.7651  0.4344 -0.2859
#> -0.3124  0.1455 -0.1687 -0.6143  0.3233
#>  0.1597  0.6859  0.2772  0.5589  0.4177
#> [ CPUFloatType{3,5} ]
```
