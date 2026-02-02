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
#> -0.5544  0.2585  0.9418  0.8673  0.3755
#>  0.8535  0.8108  1.0797  0.6063 -0.0543
#> -0.3505 -0.5354  0.8824  0.0086 -0.1262
#> [ CPUFloatType{3,5} ]
```
