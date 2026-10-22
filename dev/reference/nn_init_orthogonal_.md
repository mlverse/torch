# Orthogonal initialization

Fills the input `Tensor` with a (semi) orthogonal matrix, as described
in
`Exact solutions to the nonlinear dynamics of learning in deep linear neural networks` -
Saxe, A. et al. (2013). The input tensor must have at least 2
dimensions, and for tensors with more than 2 dimensions the trailing
dimensions are flattened.

## Usage

``` r
nn_init_orthogonal_(tensor, gain = 1)
```

## Arguments

- tensor:

  an n-dimensional `Tensor`

- gain:

  optional scaling factor

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_orthogonal_(w)
}
#> torch_tensor
#> -0.1285  0.6732 -0.4081  0.2964  0.5252
#> -0.0360  0.2841 -0.2853 -0.9111 -0.0804
#> -0.0070 -0.2918 -0.8431  0.2087 -0.4006
#> [ CPUFloatType{3,5} ]
```
