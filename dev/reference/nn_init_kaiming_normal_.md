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
#>  0.0115 -1.2448 -0.6728 -0.6933  0.3798
#> -0.0779  0.2641  0.1292 -0.0460 -0.6522
#> -1.1795 -0.3217  0.4726  0.7695  0.2473
#> [ CPUFloatType{3,5} ]
```
