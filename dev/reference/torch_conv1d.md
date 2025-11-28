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
#>  Columns 1 to 8  -3.2653  -0.9976  -1.0362 -10.7507  -1.9698   0.3753  -8.4158   9.7004
#>    1.0893   4.1370  -3.4766  -3.5308   6.8034  -1.2209  -4.3525  -7.5689
#>   -7.7431   0.8984  -1.3250  -4.9648  -7.0226   1.8597  15.3416   0.8595
#>   -2.5807  -7.4660   4.9548  -5.9534 -12.9445   1.9189   9.2976  -6.6675
#>    5.2536  -3.6797   9.5468  -1.7900  -9.3970   2.8658  -0.3788  -4.1684
#>    0.8230   6.9939   4.9063  13.4998   0.7178  -4.6755  12.4656   5.7184
#>    3.1977  -4.8000  10.3063  -9.3827 -15.8764   0.0729   5.1299  -2.8891
#>   10.8417  -1.5988  11.9862   4.1419   5.0128  -9.3476   1.0110  -4.0917
#>   -7.1320  -9.9471  -5.9383  -0.4772   2.4697  -5.1656  10.8649   2.6380
#>    5.8445  -1.9318 -10.4609   9.5357   1.7981   1.1781  -2.2447  -2.1963
#>    0.5231   8.6821   7.4119   5.1266  14.6087   4.9006   3.3383  -2.6309
#>    2.0541   1.0296  -5.0126  -7.5688   5.5584  -9.3396  -8.2599   1.2072
#>   -7.0644  -3.8127   0.5640   1.0961  10.2857  -3.5273   0.6565   7.1310
#>   -1.3456  -7.9145  15.0781 -12.3625   0.9802  -1.5301  -0.2327  -6.6282
#>    2.0784  -1.2600   1.0799   3.8569  -3.7290   0.5367   4.6456 -10.7670
#>    1.5216  18.7419   5.3334   3.6242  -0.2487  10.0502  -1.6821   3.7209
#>   15.2358   2.1414   9.1660   7.2148  11.5810  -0.7248  -4.8067  -2.9642
#>   -0.0260   7.1998 -20.8811  17.0611 -10.2380  -2.7494  -0.8996   4.0766
#>   -7.2888   5.0066   4.9187   3.9105   3.5974   9.1826  12.8047 -11.7079
#>    6.4011   4.2710  -0.1889   5.1387   0.0640   0.8079   0.0108  -2.1768
#>   -3.7626  10.4243  -1.8661  10.0367   0.1646  10.5574  -6.4387   3.0523
#>   -7.9838   3.5755   5.2327   2.2534  -3.2585  -0.9602  -3.7380   2.1328
#>   -2.4661   0.2491  -0.9027  -3.8856   7.8135   2.9552 -13.7684  -2.7443
#>    4.1565  -1.1182  18.2840  -7.4318   8.8771  11.3337   3.5615  -1.8425
#>    3.3720  -1.8663 -10.4694  -8.9599   6.3044   3.5571  -3.5527  -9.1084
#>   -3.7908 -14.0068   0.2430  -0.6857  -4.2516  -0.2145  -1.3117  -1.7840
#>   -0.3029  10.4316   3.0918   9.9733   1.6410  -1.5856   1.5658   3.2538
#>   -1.8730   4.8942  -9.0426   2.1245  -2.5930 -10.5703  -2.3237   2.2172
#>    0.4235   3.7970   0.4458   0.7460   7.1273  12.9771  -3.5685   7.8301
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
