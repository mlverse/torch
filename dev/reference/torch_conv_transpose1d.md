# Conv_transpose1d

Conv_transpose1d

## Usage

``` r
torch_conv_transpose1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sW,)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padW,)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padW)`. Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dW,)`. Default: 1

## conv_transpose1d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

inputs = torch_randn(c(20, 16, 50))
weights = torch_randn(c(16, 33, 5))
nnf_conv_transpose1d(inputs, weights)
}
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 6  4.1439e+00 -9.8365e-01 -2.7007e-01  5.7853e+00  1.6546e+00  1.3952e+00
#>   5.0665e-01  5.2420e+00 -2.3113e+01  2.8363e+00 -5.5734e+00 -4.6772e+00
#>   2.6869e+00  3.4417e+00 -3.1054e-01  1.5966e+01 -5.4684e+00  6.3421e+00
#>  -1.6954e+00  9.5544e+00 -5.8237e+00 -6.5747e+00  2.0858e+00  7.9665e+00
#>  -8.2727e-01  3.8074e+00  1.5034e+00 -4.4479e+00  4.6199e+00 -9.5070e+00
#>  -9.9540e-01  7.1938e+00  2.2686e+00  6.8795e+00  6.6197e+00  1.4498e+01
#>  -1.8405e+00  1.0570e+00 -1.2591e+00 -5.8555e+00 -1.0557e+01  9.0730e+00
#>  -3.4776e+00  1.0523e+00  1.4030e+01  7.1869e+00 -9.9857e+00 -6.2579e+00
#>  -5.0470e+00 -1.0029e+01  3.6276e+00 -6.6987e+00 -7.2419e-01  1.1434e+01
#>  -7.6382e-01 -4.2167e+00 -1.3559e+01 -1.6472e+01 -1.3297e+00 -7.9186e+00
#>  -4.0025e+00 -5.2044e-01  5.1685e+00  2.7138e+00 -2.6603e+00  2.5075e+00
#>  -5.3103e+00  1.0792e+00 -8.5519e-01  3.2305e-01  2.6344e+00 -2.2983e+00
#>  -8.8922e-01  5.7939e-01  8.5678e-02 -6.7162e+00 -1.1935e+01  4.2205e+00
#>   1.9588e+00  1.9374e+00  1.7537e+00  3.6724e+00  5.6349e+00  4.7913e+00
#>  -2.3167e+00  6.6151e+00 -6.8767e-01  1.0692e+01  1.4927e+01 -1.0599e+01
#>   6.4689e-01  1.0578e+00 -4.0603e+00  1.0319e+01  1.3215e+01 -3.4957e+00
#>   3.1129e+00 -4.5464e-01  7.8389e+00  5.4829e+00  1.0192e+01 -3.2264e+00
#>  -1.2998e+00 -1.3943e+00 -6.9997e+00  4.0970e+00 -1.3132e+00  5.6698e+00
#>   3.9403e+00 -6.7125e+00  1.6171e-02  4.6777e+00  1.6473e+01 -5.5444e+00
#>   2.1194e+00  5.8684e+00  1.2070e+01  7.8687e-01 -2.5721e+00  8.8444e+00
#>  -1.7309e-01  8.1431e+00 -2.7484e+00 -3.6758e+00  4.6816e+00  9.3894e+00
#>  -3.3191e+00 -5.3566e+00  9.4494e+00  2.3561e+00 -1.2531e+01  1.8802e+01
#>   4.2412e+00 -8.6312e-01  1.7212e+00 -4.0045e+00  7.7549e+00  7.4931e-01
#>  -2.7685e+00 -3.5850e+00  7.4757e+00  9.2712e+00  6.2834e+00 -8.8256e+00
#>  -1.5004e+00  6.5255e+00  7.9220e+00 -4.2033e+00  4.5281e+00 -1.2006e+01
#>   1.6811e+00  5.2561e+00  3.1695e+00  2.0093e+00  1.8621e+01  2.0343e+00
#>   6.8031e-01  7.6053e+00  6.8231e+00 -6.3799e+00 -2.5766e+00  1.7014e-01
#>   1.8011e+00  7.7550e-01 -9.2597e+00  1.3065e+01  3.0374e+00 -4.4932e+00
#>   4.4215e+00 -5.9739e+00  3.6703e+00  1.1059e+01  3.3021e+00  1.2241e+01
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
