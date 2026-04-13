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
#> Columns 1 to 6 -4.3283e+00 -1.3475e+00  2.9728e+00  6.9266e+00  1.2389e+01 -3.6917e+00
#>   1.1596e+00 -9.4795e-03  5.9225e-01 -8.6981e+00  5.4807e+00  7.3633e-01
#>  -1.2138e+00  3.3321e+00 -1.9893e+00  3.2945e+00 -1.0034e+01 -9.9758e+00
#>   1.2121e+00 -2.7872e+00  3.1848e+00  8.8379e+00  7.2133e+00  5.5169e-01
#>   5.8013e+00 -1.3350e+00 -1.1215e+01 -5.4821e+00  1.5556e+00 -1.0690e+01
#>   2.2183e+00  3.1410e+00 -9.1548e+00  1.4416e+00  1.2424e+01  6.2125e+00
#>  -3.2678e+00 -6.2039e+00  6.3618e+00 -4.1080e+00  7.0549e+00  1.0264e+01
#>   2.8869e+00 -1.5539e-01  7.4954e+00 -1.6807e+00  1.3288e+01  2.5187e+00
#>  -5.6104e+00  5.3766e+00  3.1209e+00 -3.6439e+00 -1.5209e+01  8.2746e+00
#>  -8.6000e-01  1.1751e+01  5.4620e-01 -8.7488e+00  2.4012e+01  1.1140e+01
#>   4.5650e+00  8.5030e+00  3.5633e+00  8.4249e-02 -3.3286e+01 -5.0860e+00
#>   1.0497e+00 -4.5600e+00 -1.3399e-04 -1.1687e-01 -5.0894e+00 -1.0713e+01
#>  -4.7794e+00  5.9760e+00  4.8792e+00 -8.2202e+00 -3.2037e+00 -3.5578e+00
#>   4.1346e+00  5.7701e-01 -2.1509e+00  9.8933e+00 -1.2630e+01 -1.3400e+01
#>  -6.8994e+00  1.0979e+00 -1.0272e+00 -1.6487e+00  1.7542e+00  7.7165e+00
#>   7.8546e+00  4.5427e+00  3.3902e+00  2.8191e+00 -1.1399e+01 -7.4789e-01
#>   7.2241e-03  4.0224e+00 -2.8430e+00  8.3228e-01 -3.9671e+00  3.7320e+00
#>   4.0811e+00 -5.1184e-01 -1.0642e+01 -1.5538e+01  2.0451e+01 -3.5959e-01
#>  -7.1129e+00 -9.3555e-01  1.6787e+01  8.5337e+00 -3.3656e+00  2.2003e+00
#>  -6.5814e-01  2.7589e+00  7.3575e+00  3.7807e+00  3.1122e+00  6.6496e+00
#>   4.3989e-01 -4.3772e+00 -2.6650e+00  6.8987e+00  7.3209e+00  1.0944e+01
#>  -4.5818e+00  6.3853e+00  7.0558e+00 -2.1127e+00  1.1705e+01  1.6781e+00
#>   3.9732e-01 -9.9072e-01  3.1925e+00  5.6395e+00 -1.3360e+01 -6.1687e+00
#>   5.6288e+00 -2.0778e+00  2.8701e+00 -8.1652e+00  1.4767e+01  7.3548e+00
#>   3.9722e+00 -6.4418e-01 -4.8581e+00  6.5826e+00 -4.7269e+00 -1.2583e+01
#>  -6.0794e-01  4.6632e-01 -2.8853e+00 -6.7317e+00  1.9556e+01 -1.3216e+01
#>  -1.4720e+00  6.3684e+00 -9.1659e+00 -4.6056e+00 -1.4526e+00  1.1856e+01
#>   2.5586e+00 -2.8326e+00 -6.0697e+00 -3.3970e+00  6.5485e+00  3.4437e-01
#>   5.7560e+00  8.0294e-01  4.5669e+00 -3.5175e+00  4.4180e+00 -6.2439e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
