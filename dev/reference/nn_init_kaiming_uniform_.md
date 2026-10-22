# Kaiming uniform initialization

Fills the input `Tensor` with values according to the method described
in
`Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification` -
He, K. et al. (2015), using a uniform distribution.

## Usage

``` r
nn_init_kaiming_uniform_(
  tensor,
  a = 0,
  mode = "fan_in",
  nonlinearity = "leaky_relu"
)
```

## Arguments

- tensor:

  an n-dimensional `torch.Tensor`

- a:

  the negative slope of the rectifier used after this layer (only used
  with `'leaky_relu'`)

- mode:

  either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves
  the magnitude of the variance of the weights in the forward pass.
  Choosing 'fan_out' preserves the magnitudes in the backwards pass.

- nonlinearity:

  the non-linear function. recommended to use only with 'relu' or
  'leaky_relu' (default).

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_kaiming_uniform_(w, mode = "fan_in", nonlinearity = "leaky_relu")
}
#> torch_tensor
#> -0.2774  0.5017  0.6114  0.9191  1.0447
#> -0.1072 -0.4474  0.1659 -0.5740  0.3206
#>  0.4924  0.0059  0.9956 -0.1081 -0.8287
#> [ CPUFloatType{3,5} ]
```
