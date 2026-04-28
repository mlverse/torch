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
#> Columns 1 to 8  -0.3559  -1.5591 -14.7554 -12.0069   2.5121 -11.6482  -3.8765   3.7316
#>   -4.3550  -7.9877   0.6015 -15.7705  -4.4273   6.8849  -3.5008  11.5878
#>   12.5757   0.4679   1.1615  -6.8723  -3.4926  -7.9029   9.1167  -7.9454
#>   -8.8246  -2.2355   3.1949  -8.9808  -4.0084   8.8133 -12.0298  -1.7613
#>    0.4739   0.2312   6.2091  -7.3042  12.2330  -3.9860  -4.1787  -2.9938
#>    6.6346  -2.8943   4.7946  14.5651  11.0521   1.3278  13.4329   3.7152
#>    8.0761  -5.6269   7.9956   3.7129   3.6107 -11.4071  -0.8799   0.3979
#>    1.4549   2.9628 -10.7399   0.7905 -10.3990   1.6644  -3.9995   9.7470
#>   -2.7597   6.7960  -5.1407  11.3881  -1.1363   1.1613   1.8515   2.7554
#>   -1.4300  -8.8141  11.4122 -17.5536  -1.4853  -0.8869  -0.2997 -18.7647
#>    1.9559  -9.1676   4.3995  -6.0887  -2.5891   0.9244   5.4575   7.3305
#>   -5.0992  -7.9267  -1.9537   5.5618  -9.5665   7.6251   0.9794 -11.5375
#>    7.3327  -0.0458  10.4261   6.0634  -3.4040  -2.1179  -1.2984  -2.0849
#>   -4.5604   1.2308  -5.5623  10.2849  -0.1902   4.0502  -0.3282  -0.5208
#>   -2.0332   5.2673  -4.3276  -4.1633   8.0692 -10.2950   5.0934 -11.5042
#>   -1.9510   0.2680  10.0087 -10.9619  -2.1149   2.2842  -5.2922 -12.4025
#>   -1.5145  10.2927  -6.5381 -10.2894   6.7788  -9.6243  -3.6183  19.6745
#>    4.0197  -5.7707   0.4338 -14.4066 -18.9076 -13.6066  -4.5708 -17.8223
#>   -1.6618   2.8679 -12.2197  -4.0447  -2.3335  -2.6872  -6.3145   5.7708
#>   -4.4148  -2.8986  -0.8887 -11.9667   4.8337  13.3051 -12.1802  -2.9565
#>   -0.0884   4.9748  -4.0585   4.8279  -7.1143  -0.4254  -3.9625   4.2624
#>    7.6065   1.3247   0.2405  12.2394 -19.6607  -7.1771   0.3424 -21.7122
#>    0.5178  -4.7000   1.9847  -4.2642   0.4833  -0.4041  14.2758  -4.8377
#>   -7.0593  -7.5946  -1.1526   3.4260  -4.3245   7.7321  -0.1755   0.4742
#>    1.6445  -1.8166   5.5843  -4.7672  -3.4306   1.3918  -1.1623  -2.8967
#>   -2.8069   6.3139  -5.4167  -0.8115   5.9926   4.4937   1.0435  13.5864
#>   -0.0688   4.8893  -2.8408 -11.3274  -3.8204  -9.0981  -7.2327  18.6435
#>    4.9028  -3.2774  -1.2739  -0.6698 -17.2085  -7.3966  -5.6024   7.0152
#>   -4.8374   1.6849   2.4658   3.4242   9.0333 -10.8108   9.0315  -9.1390
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
