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
#>  0.3886 -0.0348 -0.7705  0.5465 -0.9290
#>  0.7483  0.0976  0.2487  0.1301  0.0050
#> -0.6896  0.4241 -0.6630 -0.2552 -0.4803
#> [ CPUFloatType{3,5} ]
```
