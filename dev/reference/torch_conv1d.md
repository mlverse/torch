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
#> Columns 1 to 8   5.6824   5.7604  -2.3117   2.6387  -3.6377   5.2515  -8.6453   7.1152
#>    9.4872   2.1055   5.8310   8.9701  -3.5514  12.5988  -1.5068   5.9424
#>    9.2806  -5.7041   0.9020   0.3400  -3.3468   6.4673  -5.6933  17.7032
#>   -7.3102  -8.5663   5.9683   4.1033  -4.0464 -16.0228  -8.8675 -18.0342
#>    3.9447   3.9717   3.1260   1.8135  -0.6221  -4.5496   0.0403   7.1191
#>   -0.1834   2.7980 -11.4430   4.0214   5.9407   2.8084   1.8837   9.2262
#>    4.8986   5.8729  -2.4454   9.5599   7.1273   0.9846  -9.9290  10.2920
#>    3.0994  11.9951 -13.6084 -10.1218   4.5340   2.0061   7.9965  -0.0083
#>   -4.1153   5.0541  11.2084  -1.8727  -2.4304  -6.0018  -6.3356   2.4298
#>    9.9788  -2.1616  -3.6403   7.0974   6.2597   4.4882   3.8667  -9.9564
#>   -0.9833  -6.6841  -1.7422   1.6392   5.0394   6.8458  -8.6982 -10.8833
#>   -7.5543  -3.5783  -5.7798  -1.7921  -1.8624   0.4399  -3.8911  10.2489
#>    0.1966  -4.8974  -3.3623 -17.8482   1.1611   6.0963   8.0709  -1.2937
#>   -4.2932  -0.8821   2.0271  -3.0842  -7.0275  -3.0130  -6.2007  11.1461
#>    2.3812  -9.4406  -8.1048   2.4743   1.8534 -11.5909  -6.6818   3.7698
#>   -6.8943   0.3377  -2.2679  -3.3340  -1.9730  -4.2428  12.8984  -4.2944
#>   -8.3502  -1.7848  -6.1017  -6.8024  -0.2926   5.0344   5.4325 -13.8058
#>    2.6034   0.0784   9.9656  -3.7293  -2.0678   0.6873 -10.1898  -0.2011
#>   -3.8217   7.1772   2.4799  14.2528  11.2141   4.9187   5.8674   0.4495
#>   22.5400  -9.4935   1.8065   4.0646   0.5808   5.0631   2.5724  14.4330
#>    3.4766   4.4474  -4.3349   3.0864  -6.5971  -7.6243  -3.9098  10.3635
#>    0.3200 -14.8034   7.2691   8.0577   7.2833  -3.3490  -4.2304 -11.4389
#>  -10.1030  -1.6786   8.6498   2.9434  -1.0819  -5.7549  -1.4793 -16.9807
#>   11.2848 -12.6831   5.0564   3.5295 -11.3186  13.9411   2.4871   7.5305
#>    2.7996   0.2440   0.2721  -6.8862  -3.5029  -0.0122   3.1696   8.0245
#>    2.6959  -7.9435  -8.3660  -0.1738  -5.4128   1.0057  -7.9173  -2.8428
#>    7.2889  -1.2305   5.5476   4.4079   2.3126 -10.7005  -3.9166   8.6047
#>    2.9020  -3.0895  -3.3745   2.2927   2.0790  -8.7631   8.7193  -5.0923
#>    3.3656  -4.0784  -2.8163  -0.0380   5.8250  13.6687  -1.7244  -9.3395
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
