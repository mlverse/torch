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
#> Columns 1 to 8  -8.2169   7.8486   6.5183   2.7316  -6.8236  10.7280  -8.8263 -15.4866
#>   -4.4877  -8.8472  -0.6263   8.7725  -7.0038  -5.7449  -9.9360  -0.6269
#>   -4.3023   1.7726   5.6945   7.9512  -3.1545 -12.5909   1.5312  -0.8548
#>    2.8863   5.0463   5.1706  -7.5858  -4.7623  -6.5081   9.3586   1.6785
#>    7.4434   1.1192 -22.8756   0.2356  -5.3745  -1.4544   9.0900   1.4156
#>    0.8136  12.9006   8.3412   3.0333   2.6042   8.7850  -2.3276   1.9475
#>    3.7883  -1.1973   7.1302   0.8986  16.1355  15.7341  13.2980   2.1583
#>   -3.8951  -3.5869   4.6512   5.0763  -6.0142   0.8370  -2.1434   5.8379
#>   -7.3353   1.6979   7.5527  -1.0514   6.8811  -7.9713 -21.1118 -20.1647
#>    5.9637 -11.3959  -9.7760  -5.1533   4.3900   3.8300  -7.3052   3.3005
#>    6.8226  10.8590 -17.0681 -20.8608 -13.7849  -3.1992  -6.4364   2.4188
#>    5.2311   4.2505  11.0694  -3.0958   2.6773 -16.2052 -17.9817  -6.2020
#>   10.6949   4.6650 -11.4569  -6.0138 -11.2132  -8.3722   3.0839   2.9242
#>   12.1270   5.2298   6.0082 -17.3177 -19.7599 -16.0919   0.6275   1.7245
#>   -6.4104  15.6604   9.9547  -7.2229   4.8721  10.1523  -7.5688   0.4359
#>    0.7970  -1.3965   9.9538   7.1920  -2.6782  -1.2648   4.1648  -9.0571
#>  -10.1001  -4.0724  15.1455   5.5918  -2.8168   9.3349   5.3961  -0.3846
#>   -1.3432  -6.0129   3.0096   8.4361  -4.4921   9.6642  14.8531  -1.7767
#>    5.0015  -7.8679   6.8084  -5.0808  -1.4776   1.0681   2.1438  11.6308
#>    5.3632  -2.4056  -6.7463 -11.1144  -9.1433  15.6349   7.7211   4.0947
#>   10.6322  -8.6794  -6.1754  -5.6336   6.5793   4.8396  -3.4025  -4.1980
#>    5.4751  -5.6645  -5.3757   2.8677  -9.8050  -1.1130  11.2425 -10.6050
#>   -8.6738   5.0149   3.0255  -4.8600 -21.1054  -3.7963  -6.3260  -0.6800
#>   -3.3314  10.2320   2.7840 -13.0544  -5.6472 -15.1571  -4.5542 -23.4350
#>   -0.8726   3.6511  -2.7998 -13.6419  -0.8724   7.6622 -12.5367   2.3014
#>    2.2834  -5.2333 -11.6603  -5.1003  -1.3490   2.1135   5.1427  -6.6183
#>   -9.0493  -8.8409  -4.3720   3.9092  -1.3272   0.9113  -0.8715  -6.9122
#>    8.0807  -2.5099   0.4866  -6.7716  -8.8250   8.0276  -4.7948   1.6901
#>   -5.8618   5.6713   9.2837  -5.3438   8.7180   1.8680   1.4706   0.9341
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
