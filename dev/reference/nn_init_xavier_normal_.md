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
#> -0.7812 -0.1728  0.0844  0.5342 -0.8888
#> -0.6380  0.3295 -0.2882  0.5248  0.5069
#>  0.0882  0.0121 -0.5736 -0.2416  0.2447
#> [ CPUFloatType{3,5} ]
```
