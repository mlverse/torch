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
#> Columns 1 to 8   0.1421  -6.7754   2.7618 -20.2681  23.4928  -0.9917   0.4947   4.5237
#>   -1.1574   5.1733  -3.1739   2.6746  -7.9722   6.7556   6.6477 -12.3598
#>   -4.1507   5.5728   7.1118  -0.5085   5.5813 -23.1107  -8.7213  -1.5912
#>   -3.1752  11.6178   2.8799  -4.4582 -11.8766   3.6297  -2.8369 -14.9638
#>    1.4187  -4.6639   0.1058  -8.5844   6.2565 -11.3427 -10.3544  -3.5769
#>    5.5992   7.3490 -17.4022   6.6495  -9.2467  -7.5807   7.6137  -9.0137
#>    1.9760  -4.6826   7.9970  -9.0941   4.2294   2.5704 -11.9915   3.5820
#>    5.0147   8.1808  -2.1795   2.6218   8.2605  -9.0460   3.5842   9.3711
#>   -1.0763  -3.4868   1.0506  -5.8444  14.8999   5.7772  -8.5618  -0.5822
#>   -5.7673   8.0597 -10.8661  -5.1239  -5.4522   4.0057  17.4115  -5.0933
#>    1.3709   9.7890   1.3245   2.2252   3.7920   3.0168  -0.1751 -11.6177
#>    6.1093 -11.5573   1.5882  -5.7756   8.9593  -0.0574  -1.6324   5.5220
#>    0.8812  -0.9837   1.6076  -6.3047  -4.3946 -14.9339   3.4777 -18.1602
#>    3.4394  -1.6977  -1.4258 -11.6563   0.0221 -11.9036   0.4112  13.0106
#>    0.0032   2.3189   4.8468 -12.2698  11.4317  -9.2944   2.6775   2.4758
#>    2.8700   6.0556  -3.4716  10.9206   5.6959  -9.7796   2.1245   3.7295
#>   -4.4105   2.4811   7.3759  -9.4197   0.6515   7.7960  -0.2273  -5.2524
#>   -6.3143   9.0196  -9.7456   5.4870   1.8843 -11.1437  13.4732  -1.4543
#>    0.0546  -0.5382  17.3125 -11.6926  10.0052  -5.9304   4.1579  -3.1332
#>   -4.3909   0.6494  -8.3836   4.6565   5.9085  13.5289   4.3570  13.7761
#>    5.1791  -4.1589  -2.9407  -7.6625 -10.1584  -4.0314  19.8532   9.6506
#>   -4.6799  -0.6639  -2.9249  -0.2538  -3.3026 -10.9116 -16.0799  -7.6293
#>   -1.2944  -2.5106  -6.1178  -7.0755  -1.2758 -10.0869   8.1324   7.5569
#>    3.6256   0.6349   7.1249  -3.0349  -3.0857  -1.4959   0.0867  -8.0737
#>    1.8816   1.4196  -4.7935  13.5913 -27.1685  22.8758  -9.2317   4.6679
#>   -6.2744   2.9779  -1.0802   0.7043   8.0884  -0.3533  11.7728  -5.7876
#>   -5.1495   3.4293   2.2221 -11.9810  12.3779  -4.9427  16.0120  -2.8154
#>   -2.0687  -0.5165   4.1990  -0.7882  -3.2043  11.5668  -1.9487  13.4778
#>   -0.4363  -0.9960   8.9167   1.2658   3.8534   1.6056  -5.1743  -3.0400
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
