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
#> Columns 1 to 6 -1.8455e+00  3.0789e-01 -6.7423e+00 -5.3592e+00  1.6801e+01 -1.3469e+00
#>  -3.4914e+00  2.5706e+00  8.2115e+00  1.6072e+00  1.2736e+01  2.6622e+00
#>  -1.7986e+00 -6.8340e+00 -3.8017e+00 -4.0746e+00 -5.7523e+00 -3.1007e+00
#>   1.4054e+00 -4.2204e+00  2.7794e+00  3.6744e+00  1.2270e+01 -1.0813e+01
#>  -5.0013e+00  1.4840e+01 -9.1401e+00 -3.3290e+00  1.4593e+01 -2.1932e+00
#>   3.5633e+00 -7.5183e-01 -3.4541e+00  1.2027e+01  5.0159e-01 -6.1920e+00
#>   3.1236e-02  2.3619e+00  2.5493e-01 -1.4875e+01 -3.9619e+00 -3.3369e+00
#>  -3.2527e+00  2.8454e+00 -2.7575e+00  1.2997e+01  1.9152e+01 -8.9181e+00
#>  -4.0554e+00 -3.4485e+00  5.3169e+00 -7.7092e+00  3.1482e+00  1.5178e+01
#>   5.8896e+00  5.1097e+00  8.5850e-01 -3.1287e+00 -8.1191e-01 -5.1759e+00
#>   2.5694e+00  3.1097e+00  5.6880e+00 -5.4512e+00 -1.1604e+01  1.7988e+01
#>   4.9989e+00  4.2181e+00 -4.2875e+00  1.1391e+01  6.6584e+00 -6.7976e+00
#>   3.7150e+00  1.0063e+01  4.2890e+00  6.5196e+00 -1.8527e+00  1.4123e+01
#>   6.2674e+00 -4.3556e+00  1.2512e+01 -2.3283e+00 -8.6552e+00  4.2094e+00
#>   6.0175e+00  3.2912e+00 -4.6517e+00  1.5573e+01  1.0566e+00 -1.1967e+01
#>   7.9220e+00  3.7932e+00 -1.6306e+01  6.1921e+00 -1.7661e+01 -1.9039e+00
#>   1.4952e+00  3.5796e+00 -4.4448e-01 -1.2503e+00 -1.6956e+01  7.4638e+00
#>   6.9434e+00  3.1447e+00 -7.4632e+00 -6.2554e+00 -6.8206e+00 -4.2835e+00
#>   2.7315e+00 -1.9676e-01  9.6763e+00 -7.0368e+00 -5.5796e+00  1.9574e+01
#>  -6.8456e+00  7.6375e+00  9.5059e+00  7.9380e+00  5.3483e+00  7.1094e+00
#>  -1.8581e-01 -1.5129e+01 -6.8813e+00 -1.1580e+01 -6.2615e+00  3.9596e+00
#>   6.8704e+00 -1.2620e+01 -5.8258e+00  1.9673e+01  2.1307e+00 -1.1478e+01
#>   3.2431e+00 -6.0341e+00 -2.7493e+00 -3.9606e+00 -2.4457e+00 -1.3221e+01
#>  -4.7273e+00 -4.9133e+00 -5.8002e+00 -1.6554e+00  1.0053e+01 -6.0304e+00
#>   4.6522e-01 -1.4823e+01  5.2965e+00  7.7242e+00 -3.3479e+00  4.4198e+00
#>  -3.5364e+00  4.9180e+00  1.6208e+01  1.1054e+00  2.8610e+00 -3.6337e+00
#>  -2.4256e+00  7.0506e+00 -1.0381e+00 -6.6683e+00  9.2550e-01  3.3400e+00
#>  -1.7520e+00  1.3224e+00  7.4710e+00 -2.9390e+00  6.8627e+00  2.4129e+01
#>   5.5641e+00  2.5968e+00  6.7009e+00 -1.8161e+01  1.7616e+00  3.7029e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
