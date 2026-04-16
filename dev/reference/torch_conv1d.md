# Conv1d

Conv1d

## Usage

``` r
torch_conv1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv1d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See
[`nn_conv1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

filters = torch_randn(c(33, 16, 3))
inputs = torch_randn(c(20, 16, 50))
nnf_conv1d(inputs, filters)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8 -17.1948  -6.7704  -0.0844  -0.7669   6.2164   0.3905  -0.6225   0.5624
#>  -12.8746  -1.1036   6.1799  -1.2108  -0.0312   3.6315  -9.3341   8.1450
#>   -1.4902  -1.3015  -4.4837 -10.2621   7.6343  -6.0148  -3.7874  -3.5812
#>   -1.0060  13.8829  10.4751   7.7364  -0.5894  -4.9648  -7.5849   4.6091
#>   15.2610  -1.2953  -2.1634  12.1115   3.6633   1.9027   5.0017 -13.1099
#>  -10.1501  -3.4253 -25.0827  -2.1641   2.4552  -2.1528  13.5406 -23.6483
#>   -2.3938  10.4140   4.8704  13.8811  -1.4182   7.6550   1.3756  -4.0348
#>    7.2756   4.0987   7.0639   2.6928   6.8058  -3.6877  11.3666   3.8336
#>   -2.4563  -0.3030  -3.7109  10.8475  -2.5258   4.9001  -4.1066  -5.1673
#>    3.8085   5.1368   3.8510  -5.8824  -6.0668 -12.9823  -4.6266   4.2142
#>   -3.3151   0.3498   3.6375  -4.2785  -8.8590  11.7050 -15.0437   2.7053
#>    8.8679   2.7421   3.5171  10.2294   2.0283  -1.5700  -2.1079   0.1104
#>    2.3360   3.3662  -0.7228  -0.1713  -8.8305  -5.7781   4.1214  -4.6128
#>   -9.1529  -2.6382 -15.3839  -2.3492  -6.4064   4.9352 -14.1183  -5.0461
#>   -8.3714  -0.2154   9.3774  -3.9195  -3.7229  -0.6026  -6.2927  -2.1817
#>    4.6962  -6.6598  -5.9528  -6.8683   3.0025  -7.5311  12.1430 -13.9176
#>  -18.4202 -11.2478  -5.4753  -1.5299  -8.5818   0.8036  -4.4520  -1.6596
#>  -11.1655   6.7406  -0.9209  -4.8074  -2.4845   4.8763   1.5245  -2.9432
#>    5.5919 -17.5732   1.0129  -2.4905   2.7402  -4.2185 -16.6613  -2.0710
#>   -6.0985   7.1692  -3.4625 -13.9143   2.7404 -12.3766  20.4215   2.6173
#>    1.7260   3.4279   7.1875  -3.9625   7.9832  -2.9589  -0.8797  -5.7123
#>    8.6476   0.7825   5.7134   2.7209  10.4062  -9.3447   1.8921  -7.0463
#>    9.4103  -2.9540   3.3663   3.6432   1.5948  -3.3500  -5.6357  -3.0818
#>   -4.0205  -7.9615 -16.3524   1.1092  -9.6020  -6.2959   8.1313  -2.7126
#>   -2.1145   3.6778  -0.7622   4.6580   2.5025  -0.2445  -0.0708  -4.8230
#>    2.3814   5.7010   9.2443  -2.3314   1.0742  -3.0035   9.2062   6.3063
#>   -9.6411   5.0235 -11.5796  -9.0412 -11.1752   8.7382  -3.6292  14.5226
#>   -0.7354  -4.2448   3.4884   3.4857  -2.7642  10.9176  -8.3737   1.6819
#>    1.8529 -14.9495  -6.9398   3.0483  -4.6676   2.3889   1.5083  -0.8215
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
