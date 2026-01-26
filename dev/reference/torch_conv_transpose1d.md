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
#>  Columns 1 to 6 -3.3276e+00 -1.2624e+00 -6.1147e+00 -1.2086e+01 -1.1683e+01 -6.2908e+00
#>  -4.2096e+00  1.0706e+00 -2.7620e+00 -1.1807e+01 -3.4628e+00  9.0193e+00
#>  -2.6080e+00 -1.6150e+00 -6.4071e+00 -5.6368e-01  7.8587e+00 -1.6201e+00
#>   1.6351e+00  3.6302e-01  5.9527e+00 -3.8392e+00  5.5184e+00  1.0316e+01
#>   2.5416e+00  1.8214e+00  5.5066e+00  8.0569e+00 -4.9533e+00  3.3669e+00
#>   7.8788e+00  4.4083e-01  4.8543e-01  5.0833e+00  1.5732e+00 -5.5170e+00
#>  -4.4391e+00  1.4642e+00 -8.1877e+00 -4.5001e+00 -4.5836e+00 -4.1183e+00
#>  -1.7028e+00 -3.9960e+00 -5.4727e+00  8.3768e+00 -2.5230e+00  1.2376e+01
#>   7.9886e-01 -4.6379e+00 -4.4289e+00 -8.1100e+00  9.1593e+00  4.6666e-01
#>  -6.3006e+00 -1.0088e+01  1.8929e+00  1.3730e+01 -1.2481e+00  5.0879e-01
#>   9.2711e+00  5.7334e+00 -4.6668e+00  6.2426e+00  8.9815e+00 -4.9751e-01
#>  -2.3931e+00  8.6783e+00  1.2230e+01  2.7477e+00  9.6900e-02  4.8895e+00
#>  -4.6995e+00 -1.2548e+00 -4.4900e+00 -8.0314e+00 -1.4281e+01 -4.5927e+00
#>  -1.9788e-01  1.9347e+00 -4.5409e+00 -2.9590e-01  3.7164e+00  1.1099e+01
#>   9.2981e-01  6.2747e-01 -2.7972e+00 -2.6022e-01  1.5727e+01  9.1353e+00
#>  -7.5323e+00 -3.6714e+00 -3.2720e+00  4.7387e+00  3.7348e+00  1.3993e+01
#>  -5.3973e-02  8.1628e+00  1.5311e+00  6.6750e+00  2.4246e+00  4.1994e+00
#>  -3.3720e+00 -4.0062e+00  7.5350e+00  2.1264e-01 -2.8809e+00 -8.6576e+00
#>  -9.8408e-01  3.9340e+00 -8.3611e+00 -1.0655e+01 -6.2723e+00 -6.6047e+00
#>   3.1701e+00 -3.0323e+00  1.5873e+01  5.3080e+00  2.1348e+01 -6.8728e+00
#>   1.1057e+00  9.0911e+00  1.0714e+01  9.1968e+00 -1.7258e+01  3.5230e+00
#>   5.7500e+00 -3.5417e+00 -1.4732e+01 -5.4461e+00  9.5020e+00 -3.4832e+00
#>  -1.8652e+00 -1.0285e+01  7.0167e+00 -5.9681e+00 -1.0352e+01 -6.1688e+00
#>   9.6155e+00 -2.6872e+00  6.4569e+00  8.8342e+00  7.5453e+00  5.7954e+00
#>   2.6748e+00 -5.4617e+00  2.0263e+00 -4.9361e+00  3.5665e+00 -2.8506e+00
#>  -3.1999e+00  2.7906e+00 -2.2941e+00 -1.4337e-01 -3.4978e+00 -1.9197e+00
#>   1.4565e+00 -2.1595e+00  9.5007e+00 -1.0784e+00 -6.7896e+00 -5.1535e+00
#>   7.0681e+00 -3.3928e+00  3.8561e+00 -7.5546e+00 -2.4378e+00 -3.5859e+00
#>   1.0511e+01  1.4205e+00  3.7668e+00 -4.5874e+00 -3.7174e+00  1.2557e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
