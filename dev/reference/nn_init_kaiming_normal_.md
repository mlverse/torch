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
#> -2.0076 -0.3694  0.0962  0.3471  0.0833
#> -0.9279  0.7067 -0.6754 -0.4325  0.5376
#> -0.4489  0.8885  0.4814 -0.1413 -0.5579
#> [ CPUFloatType{3,5} ]
```
