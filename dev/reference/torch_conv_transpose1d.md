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
#> Columns 1 to 6  1.9620e+00  1.3621e+00 -7.8533e-01  3.1415e+00  4.7585e+00  1.0894e+01
#>  -3.7727e+00 -1.5492e+00 -1.5947e+00  1.0633e+01  1.2843e+01 -6.5622e+00
#>  -2.0144e+00 -1.7095e+00  2.3082e+00  1.1092e+00 -5.4906e+00  8.0543e+00
#>   3.0961e+00  7.7732e+00  4.8286e-01 -1.4592e+01  3.0112e+00  1.1468e+01
#>   2.6156e+00  4.0152e+00  1.1891e+01 -3.3313e+00 -7.5738e+00 -2.2606e+01
#>   1.3716e-01 -4.9869e+00  8.6928e+00 -3.4162e+00  6.6546e+00  5.7595e-01
#>  -2.3430e+00  3.7791e+00 -4.1426e+00  4.8924e+00  9.6375e+00 -6.4277e+00
#>  -1.2987e-01 -1.0383e+01  1.3978e+00 -2.1919e+00 -3.7338e+00 -1.1017e+01
#>   2.4728e-01  4.8514e+00 -9.0101e+00 -3.0136e+00  8.1958e+00  1.4979e+00
#>   1.2781e+00  6.4742e-01 -3.0663e-01  9.2349e+00 -4.6174e+00 -1.2922e+01
#>  -6.8139e+00 -1.4138e+00 -1.2277e+00  4.0298e+00 -2.2715e+01 -7.4322e+00
#>   3.2400e+00  1.2409e+01  8.4930e+00 -1.5593e+00 -2.3483e+01  2.0320e+01
#>  -6.9326e+00 -7.5872e+00 -2.3747e+00  1.1220e+01 -5.2316e+00  6.2476e+00
#>  -2.1009e-02 -6.1558e+00  9.8614e+00  6.7045e-01 -1.0460e+01 -6.3517e+00
#>  -1.8613e+00 -4.2637e+00 -3.3963e+00  1.0392e+01  7.3143e+00 -1.3097e+01
#>   3.1505e-01  9.5456e-01  6.4547e+00  1.3930e+01  6.6354e+00  4.9150e-01
#>   3.9622e-02  5.4054e+00  1.2501e+01  9.1137e+00 -1.4126e-02  1.7416e+00
#>   3.7717e+00  9.1910e+00 -9.7383e-01 -2.4961e+01 -7.6957e+00  3.1867e+00
#>   3.2812e+00  7.1050e+00  6.1838e+00  8.0086e+00  1.0742e+01 -5.8473e+00
#>  -6.1040e-01 -5.4834e+00 -3.6602e+00 -7.7334e+00 -3.2200e+00 -1.5752e+01
#>  -1.5480e+00 -9.5400e+00 -1.0038e+01  9.4088e+00 -4.8727e+00 -5.4677e+00
#>  -5.1203e-01 -4.8521e+00  5.4240e+00  1.2108e+00  7.8241e+00 -2.5964e+00
#>  -4.4406e-01  2.7579e+00 -1.0694e+00  7.2973e-01 -9.4174e+00 -8.4488e+00
#>  -4.5078e-01  2.7351e+00  2.0542e+00 -4.6884e+00  1.1493e+01  9.6070e+00
#>  -6.4577e+00 -8.2014e-01  6.2382e+00  1.3359e+01 -1.7999e+00 -8.7578e+00
#>  -1.8081e+00 -2.2204e+00 -1.2183e+01  5.0312e+00  2.6122e+00  1.4414e+01
#>  -2.8065e+00  8.3200e+00  1.9002e+01  1.4424e+01 -8.5323e+00 -5.4401e+00
#>   2.3180e+00  1.8498e+00  3.2091e+00 -4.8336e+00 -4.6178e+00  7.2174e+00
#>  -1.5769e+00 -4.2597e-02 -1.0792e+01  3.0089e+00  5.1880e+00  3.3447e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
