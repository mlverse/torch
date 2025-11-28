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
#>  Columns 1 to 8  -1.2045   2.1746  -6.0406  -9.4542   0.0556   3.3057 -21.7346   1.2658
#>   -6.0286   3.0178  -0.1214   4.0001  -2.5674   1.6095  -7.3186  -1.9766
#>   -5.5661  -9.5994   4.5329   2.8354  11.0229   0.0995  14.3583  10.7310
#>   -0.1365  -5.9858   4.9898  -8.6956 -12.3336  22.9718   5.3522   5.3074
#>   -1.0961  -5.6485   5.6736 -14.0752  10.1656  -6.9646   3.5731   1.0606
#>   -0.5214  -5.1442  -7.2186  18.5069   1.6767   3.7562 -10.1506  10.9495
#>    2.2260   2.6683   7.3085   7.6112   2.4122  -1.6366   7.5414 -10.2638
#>   -2.2725   1.5275  -3.6537   2.5604   8.2963  -2.5367   9.5031  -9.3216
#>    3.5435   0.5236   0.1892 -15.9096   0.6701  -1.7014 -11.0473  -5.9580
#>    3.0320  -2.0484   5.9724  -6.0568 -15.0749 -13.3584  -5.8439   0.7905
#>   -1.6898   3.2004  -2.2872  -9.9234   0.3202  -2.8561  18.4730  -2.4771
#>    3.2281  -1.0645  -8.7695   7.7103  -5.2645  -4.3046   8.8517  -9.8140
#>   -0.1994  -5.1789  -9.3687   3.6161   7.0521  -3.4444   8.7281   0.0096
#>    4.2870   4.0863   6.6476   8.3996 -15.4104  -6.3289   6.0805   9.7433
#>    4.1785   3.0143   7.6329 -10.3703  -0.8735   0.2999   6.2345   2.5947
#>    2.3780 -11.0137  -5.1327  -0.7451   8.5394  14.7211 -11.9304   8.6028
#>    0.8552   4.7810  10.6228  -0.3854  -6.1463  -3.7507  -9.5041  -0.0234
#>   -2.4715   4.3994  -4.2168  -5.4721  -6.1150 -16.1625  -8.1385   9.1491
#>    1.5212  -7.2566   7.0392  -8.1946 -11.0461   7.9210   0.4422  14.3543
#>   -2.8414   5.8389 -13.4773  13.2995  -4.5464   7.6498   5.2297  -6.5026
#>    4.9094  -1.8422   7.5222  11.3059   8.5833  -9.2854  -8.8165  -1.2594
#>   -1.3792   5.9938  -0.4012  -9.8151   8.8765  -1.5580   6.3553   0.5473
#>   -0.2313   3.8961  -8.8221   4.4597   5.8316   1.8069  -6.6280  10.0232
#>   -1.5228  -2.8653  -6.2910   6.6627 -14.4556  10.7699   8.7480   6.3909
#>   -2.1195  -1.5168  -0.1983  -4.6694  -0.6762   8.1426  -3.8480  -4.1912
#>   -0.8442  -2.2326   4.5474   2.3567  -6.5051   1.5487  -3.9088  -2.7904
#>   -2.2872  -6.6897  -1.4125  -5.3281   4.1138   2.7452 -11.7724   1.8322
#>   -0.8292   7.7085 -12.2825   2.4028  -3.4375   1.6849   1.6247 -12.2050
#>   -2.1816  -2.8016   9.1740   2.1122   3.6621   7.1870   0.6096  18.2426
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
