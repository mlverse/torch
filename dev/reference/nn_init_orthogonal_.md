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
#>  0.5132  0.5171 -0.6694 -0.0021 -0.1453
#>  0.0428  0.3255  0.3919  0.6946 -0.5061
#> -0.5055  0.3935  0.0328 -0.5565 -0.5281
#> [ CPUFloatType{3,5} ]
```
