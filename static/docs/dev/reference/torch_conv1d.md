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
#> Columns 1 to 8  -0.8376  -4.4108   5.0271   2.1678  -8.5521  16.1129  -6.7126 -12.7427
#>   -0.1683   3.4827   5.1476   9.2742  -4.6294  -1.6349  -1.3280   2.9061
#>   -0.6405  -5.3022   1.4463  -2.0851  -4.2276  -1.9461   5.0062  -9.9919
#>    9.5056   1.0239  -9.4033   7.8033   0.1651  -3.4399   5.7280   7.5863
#>   -3.3241  -7.4364   0.4105  -9.5664   6.8114  -3.8751   0.0047  -5.3302
#>    4.1859  -4.6193  -0.2414  -3.6832   5.8408  -7.9721  -4.7837   6.7049
#>    2.9281   6.2945  -0.3895  -3.3141  -5.1124  -1.0920  -9.8086   3.6712
#>    9.2107   2.2119   0.4271   0.1878   7.4234 -11.6072  -1.2127  -7.2189
#>   10.3926  -0.7828  -9.5518   2.2289   4.2855  -7.7951   1.7418  -5.5377
#>   -0.0590  -1.9113 -10.7225  -1.6611 -11.1280  -3.8062   3.9973  -4.9965
#>    8.8645  10.4985  -4.7469   2.5927   1.6868   1.5545  -3.9865  -3.7601
#>   -4.5247   2.9656  -5.2118   6.5239   6.1042  -2.9796   2.0920   6.5172
#>   -0.0894  -5.8443  -4.6824  -8.6534   7.8308 -10.6134 -10.1316   7.2305
#>   -5.0057  -0.8731  -4.4719   5.9009 -16.2739   6.5200   6.1428  -1.8664
#>    5.5184   7.4832   4.6613  -6.1343  -2.7799   4.5046  -6.1264  17.7578
#>  -10.0535  -1.6456   2.5888  -2.6252   3.9528   2.0237  -2.1517  -4.6862
#>    1.9136  -7.4350   7.9854  -2.1539  -2.4383 -18.5132  10.6547 -10.3904
#>   -1.2068 -11.4379   1.0090 -14.7205  12.4430  -1.4044   9.5179   0.6280
#>   -2.1972  12.1673   0.2380  -4.7576   2.4652  -4.1819  -8.8824 -10.7467
#>   -2.8480 -14.4337 -10.1569   7.3568   6.8843  -4.0397  -2.4893  -3.2108
#>    1.4905   4.4083  -2.0087   3.4039   9.5917  -3.1995  -0.9536  -1.7041
#>    3.7815  -1.5093  -3.6395   9.3607  -7.0079  10.7300   4.2670   7.0893
#>    1.6009  -6.0013   3.2304  -4.8344  -0.9086  12.2907   7.3559   3.9263
#>    2.9942  -1.3638   6.4839  -5.3632   3.0239   0.4088  -1.9672   2.8559
#>   -7.8053  -2.1507   1.5243  -6.1143  -0.3364  -2.5535  -4.1544  -7.1911
#>   -0.6733   7.2515   7.5055  -2.7996  11.4870  -0.0696  -0.3381  -5.0557
#>   -3.2856   4.6158  -2.4008  -1.1696  -2.6077   6.4766  -0.0433   1.8553
#>   11.4264   8.5156   4.3147   2.0917   3.9337  -4.4348   9.6038  -0.5376
#>    4.9966  -4.4202  -4.3740   1.0583   7.6228   0.0108   8.7402  -9.1712
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
