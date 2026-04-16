# Kaiming normal initialization

Fills the input `Tensor` with values according to the method described
in
`Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification` -
He, K. et al. (2015), using a normal distribution.

## Usage

``` r
nn_init_kaiming_normal_(
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
nn_init_kaiming_normal_(w, mode = "fan_in", nonlinearity = "leaky_relu")
}
#> torch_tensor
#> -0.0110  0.4636 -0.0930  0.3699  0.2507
#> -0.5921 -0.1236 -0.8601 -0.8059 -0.3438
#>  0.0278 -0.0735  0.0868  0.3404  0.3005
#> [ CPUFloatType{3,5} ]
```
