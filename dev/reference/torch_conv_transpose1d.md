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
#> Columns 1 to 6 -9.2246e+00  4.9530e-01  2.9194e+00 -1.0342e+01 -1.3261e+01 -1.0933e+01
#>  -3.7841e+00 -2.1062e+00 -1.0317e+01  6.6554e+00  4.4623e+00 -7.2459e+00
#>  -1.7232e+00  3.4690e+00 -5.0288e+00  7.1310e+00 -9.9402e+00  5.4974e+00
#>  -4.2452e+00 -2.5419e-01 -3.8222e+00 -1.8134e+01 -6.9152e+00 -4.2016e+00
#>  -4.1633e-02  4.2234e+00  3.3583e+00  5.8115e+00 -1.6574e+01 -7.5145e-01
#>  -7.3805e-01 -3.2534e+00 -1.3342e+00 -6.0060e+00 -6.6590e+00 -3.3202e+01
#>  -8.8324e-01 -3.0044e+00 -5.2761e+00  1.2753e+00  1.4280e+01 -9.2851e+00
#>  -3.8287e+00  5.8483e+00 -1.2895e+01  5.0207e+00 -2.0454e+00  9.7872e+00
#>  -1.9835e+00 -1.5108e-01 -1.2362e+01  8.5420e+00 -1.3787e+01 -1.5468e+01
#>   5.2206e+00 -4.4220e+00  1.2749e+01 -1.5448e+01  9.4168e+00 -4.2391e+00
#>  -6.1760e+00  1.1604e+01 -5.9681e+00  9.2856e+00  4.1839e+00 -2.6881e+00
#>  -3.4552e+00 -7.9310e-01 -3.5951e+00  1.8616e+00  7.3856e+00  4.6390e-01
#>  -1.5751e+00  1.3830e+01  3.4411e+00 -2.5308e+00 -8.5907e+00 -2.0312e+00
#>  -8.1983e+00  1.2309e+01 -4.3069e+00 -5.7257e+00  1.0224e+01 -7.0624e+00
#>  -7.8218e+00  2.6368e+00 -7.1724e+00  8.2115e+00 -2.1428e+00  1.2839e-01
#>  -3.3750e+00  3.3683e-01 -1.3213e+00 -3.7377e+00  1.7815e+01 -9.6427e+00
#>  -8.1670e-01  8.6323e+00  7.1082e+00  5.8932e+00  3.3795e+00  7.5918e+00
#>   4.3649e+00 -6.1800e+00 -1.0166e+01 -8.4395e+00 -1.0195e+01  3.6185e+00
#>   4.9203e+00 -3.6800e+00 -5.1976e+00 -8.9544e+00  6.2609e-01 -7.7414e+00
#>   1.4940e+00  1.6392e+00 -2.0007e+00  1.6548e+01  8.7112e+00 -7.8705e+00
#>  -8.0927e+00  6.9345e+00 -1.3261e+01  1.3711e+01  1.5960e+00  3.0832e+00
#>  -7.8079e+00  4.5284e+00 -9.2871e+00  1.7547e+00 -1.1380e+01  1.7488e+01
#>  -3.1597e+00 -1.3698e+00 -9.8269e+00  1.5091e+01  4.2069e+00  5.0501e+00
#>  -1.8927e+00 -1.2928e+00 -6.1926e+00 -1.1796e+01  6.2284e+00  1.3705e+01
#>  -4.3698e-01 -1.1027e+01 -8.2244e+00  1.1238e+00 -5.2376e+00  5.7115e-01
#>  -1.7324e+00  8.9489e+00  1.2790e+01 -2.5119e+01 -8.6472e+00  1.9653e+00
#>  -3.4198e+00 -2.3409e+00  2.9864e+00 -1.0447e+01  5.1433e+00 -9.1925e+00
#>   4.8403e+00  5.3526e-01 -1.7770e+00 -2.6633e+00  5.1170e+00  6.3532e+00
#>  -5.2883e+00  1.2745e+01 -6.2653e+00 -2.1026e+00  1.2066e+01 -2.2198e-02
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
