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
#>  Columns 1 to 6 -3.9186e+00 -3.1114e+00  5.3237e+00  5.7435e+00  9.5099e+00 -2.3142e+01
#>  -4.0696e+00  4.0659e+00  3.1969e+00  3.2851e+00  6.5325e+00  1.6838e+01
#>   1.4009e+00  2.8062e+00  3.1501e+00 -5.4773e+00 -2.5498e+00 -1.8758e+00
#>   1.6164e+00 -3.6549e+00  1.9533e+00  1.2284e+01 -1.3540e+00 -9.0780e+00
#>  -1.9815e+00  4.5327e+00 -4.1473e+00 -8.1476e+00 -7.5491e-02  1.6366e+01
#>   8.6234e-01  1.9343e+00  5.2577e+00 -8.7885e+00 -2.4610e+00  6.2475e+00
#>  -3.9013e+00 -2.6130e+00  4.4385e+00  5.8058e+00  1.1554e+00  1.1602e+01
#>   2.4626e+00  1.0427e+01  9.4464e-02 -3.5069e+00  3.2065e+00 -1.8522e+00
#>  -2.8989e+00  8.5656e+00 -5.3457e+00 -7.5939e-01 -1.4335e+01  1.9171e+00
#>  -1.7451e+00  6.2818e-01 -2.8126e+00 -9.9578e+00  5.2330e+00 -1.6111e+01
#>   5.5794e+00  3.7172e+00 -1.0889e+01  7.3195e+00  3.3373e+00  7.6144e+00
#>   3.6624e+00 -8.7982e+00  3.5188e+00 -1.5482e+01  2.0135e+00  5.4028e+00
#>  -7.7783e+00 -3.1557e+00 -3.7231e+00  6.5715e+00 -3.7112e+00  1.3124e+01
#>  -5.0794e+00 -1.6245e+00  7.3385e+00  1.8340e+00  1.8751e+00 -4.1061e+00
#>  -1.8747e+00  5.2879e+00  5.5448e+00 -1.6744e-02  1.9721e+01 -5.7797e+00
#>  -1.9671e+00 -3.5412e+00 -4.6264e+00 -1.7267e-01  1.6635e+01 -8.5389e+00
#>  -1.7311e+00  3.6001e+00 -8.0889e+00  1.6670e+00 -2.2692e+00  1.0248e+01
#>  -5.1423e+00 -2.9591e+00 -4.9535e+00  5.7053e+00 -2.7087e+00 -3.0597e+00
#>  -3.6709e+00 -5.8553e+00 -6.6570e+00  1.4688e+01  3.4441e+00 -7.4219e+00
#>   2.2473e+00  6.1215e+00 -6.6395e+00 -1.8954e+00  5.2778e-01 -1.5510e+01
#>  -2.8064e+00  4.8298e-01  8.9550e+00  4.9995e+00 -1.2178e+01 -1.2624e+00
#>  -9.0166e-01 -3.5990e+00  1.1312e+01 -6.5314e+00 -6.0373e-02 -3.1054e+01
#>   5.4112e+00  7.2771e+00 -3.4602e+00 -7.3348e+00 -1.4459e+01  9.0282e+00
#>   2.1791e+00  3.7258e-01 -1.0874e+01  1.2705e+01  5.5411e+00 -4.2561e+00
#>   2.7470e+00  1.2876e+01  8.6998e+00  6.9087e+00  3.5420e-01  5.4795e+00
#>  -3.5358e-01 -7.4106e+00 -2.4005e-01  4.8412e+00  3.4035e+00 -2.1783e+00
#>   4.5644e-01  1.1017e+00 -4.3423e+00 -2.3967e+00  8.4056e+00  1.6994e+01
#>   2.9981e+00 -1.5882e+00  1.0373e-01  1.2184e+01  1.7401e+00 -4.2873e+00
#>  -3.5223e+00  4.4900e-04  5.1215e+00  1.9800e+01  8.8730e+00  1.2971e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
