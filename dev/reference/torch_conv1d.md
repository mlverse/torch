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
#>  Columns 1 to 8   8.5066   5.2460  -5.0702   9.7661   7.8437   7.6397  -4.9277  -7.4332
#>   -6.6144 -13.2427  -4.0484  -7.4242 -12.5976 -14.2367  -9.3327  -0.1520
#>   -4.7191   7.4360  -9.3186   8.1276  -8.6746   5.7331  -5.7990  -5.0142
#>    9.9508   6.6418   4.0012   3.0726   0.4566  10.7590   4.7616  -8.2945
#>    7.1707   0.9770   7.4964   7.1305  11.7823   0.6940  -4.5239  -3.3830
#>    2.6888   3.6749 -12.3774   2.5609   1.2296  -4.5380  -9.8231   1.4923
#>    5.3915  -1.9385   3.6329   1.8798   1.1323   8.1682 -12.3050  11.0582
#>  -11.6445   8.5847 -11.3606   6.7939  -5.7865   4.6098   6.7900  -4.4457
#>    8.0151  -4.9409   3.4025  -6.7246   5.3516   5.4950  -7.3852   3.6879
#>   -4.5881   4.3745  -1.8664   8.7960   1.3976  -4.6902   4.7306  -2.1597
#>    0.9971  -4.9432   1.4991   0.3251   7.4873  -8.6693 -11.6045   3.8503
#>   -7.2399   9.6449  -4.4143   4.4314   3.0613   2.5039   7.8003  -6.3692
#>   -1.8625  -9.8309  -7.4603  -3.7788  -6.5241  -9.8672   4.6000  -1.2128
#>   -3.9187  -5.9118   8.3398   0.4095   3.7267   0.4764  -3.7722   9.7326
#>   18.7414  -0.4309   0.8262   2.5802   3.2783   5.8421  -3.6483   6.4690
#>    1.1317  -2.9794   5.7389  -1.8460   2.7296  -6.7621  -4.3571  -0.2694
#>   -4.0859  -1.1787 -11.2076  -1.9307  -6.9711   1.9999   5.1074  -4.7967
#>    7.4927  -2.9980   4.6417 -16.7079  -2.2785  11.5917   7.2102  -1.2021
#>    0.2247 -19.6565  14.3427  -0.3295   1.9919  -2.1295  -6.5629   7.6262
#>    2.8063   9.0376  -6.1869  10.7453   7.2853  -3.3743 -11.3036   0.2057
#>  -19.8373  10.7833  -5.9312   5.5679  -9.3279  -4.3200  -0.3028   0.6281
#>  -10.7551  -7.3069  -5.4449  -9.9948  -1.3125  -4.7322   0.2011   1.2233
#>  -13.0984   9.1607   4.0474   6.8254   3.6255  -1.9472   9.0260   0.4592
#>    8.1648  -3.5669   5.7597  -6.5665  -1.2986  12.2016 -12.1307  -0.3799
#>    5.7839  -2.5094  -0.4884  -2.8138  -6.3244 -11.9357   0.4833   4.6153
#>  -13.5204   6.4655   4.8186   5.6089   7.6762   7.1157  16.7533   3.2583
#>  -15.1425   2.7031   2.6893  -1.6723   3.8414   4.3749   2.0345  11.9733
#>  -10.8651   6.3996 -12.6554   3.1680  -6.7817  -3.1557  -3.0154   0.5853
#>  -12.4104  -4.9803   8.4476   9.4836   9.0326 -12.4321   3.4903  -2.2123
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
