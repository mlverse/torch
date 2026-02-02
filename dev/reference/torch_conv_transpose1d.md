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
#>  Columns 1 to 8  -6.4582   5.3984  -7.7317  -7.3641  10.0488  -7.1526   3.1397   1.7582
#>   -3.1781   3.7032  -3.3928  -2.6158   7.4310  -0.8229  -3.6056 -13.3834
#>   -1.7337  -1.9502  -1.9452  -1.4521  14.6862  -0.0873   3.8300  -2.1273
#>   -4.5048  -1.8721   2.0219   0.4052 -16.3178   1.7800   4.2103  -5.5562
#>    0.5587   0.5829 -11.5756  -7.7340 -11.4740   2.9685 -12.3602  -5.0577
#>  -10.3512  -2.2788  12.3441 -11.2390  -6.5934   6.2719   0.6435  -5.5142
#>    5.2567   1.2068   3.1129  -9.4139 -10.5536   4.3426  -3.6459  -7.4797
#>    2.7028  -7.0889   5.9796   1.3000   4.7154   1.0203  10.3273   5.0857
#>   -4.8992   1.5529   0.5765  -8.1713  -5.1314 -13.7904  -7.2034   3.9603
#>   -2.7960   8.6062  -8.4766  -4.4563  13.0977   0.6409  -9.5326   5.0057
#>   -4.5082  -4.4128  -9.5493  -1.5221   5.8928   0.4497  -4.0000   7.6490
#>    1.3046  13.2036  -6.5061   2.3444   0.6530   2.4499 -13.3151  -6.0241
#>    7.7148  -7.9908  -6.0898  -1.7193   1.8095 -12.9107  -0.9819  11.7207
#>   -1.4639  -1.1376   3.7859 -11.0191  -8.8683   8.8224  -3.9357   7.7081
#>    2.5497   0.1012  -7.5309  -5.0661   1.9959   3.2906 -12.7104   3.8993
#>   -1.2280   4.1162  -6.5877 -10.3286  23.6941   9.9508  11.7709   8.4233
#>    4.1685   3.5176  -9.6033   2.0854   1.1994  -9.2163   2.7032  -7.9569
#>   -2.5750   6.9708  -2.6871 -22.0581   0.4336  -0.2809  -2.9132  13.2829
#>   -1.1048  11.7253   0.5004   1.0824  14.2144 -17.5186  14.6769  -5.5902
#>   -8.6113   4.1394  10.3931 -15.0401  -7.2255   2.3446   4.4657  -5.9805
#>   -2.5250  -9.6034   5.1613  15.2232  -7.0162   0.2252   4.9616  -0.5477
#>   -0.9198   3.6692  -4.1081   5.0013  -6.2150  -6.1737   5.0429   6.4498
#>   -1.6102  -4.4380  -5.2479  10.1525 -11.5327 -13.6852   0.4340   6.6019
#>    2.6081   3.0765  -6.2972   2.0415  -6.0401  10.2883  -2.7104 -11.7503
#>    9.4811 -10.7802   7.2427  11.4935   5.8809   6.4511   4.2083  -7.0626
#>   10.0932   2.4992  -6.7061  -3.2769  -4.5820   6.3141  -7.6701  -3.7529
#>   -1.8357   8.2477  -0.7743  -4.4133 -10.7908   3.9202  10.9087 -10.1138
#>   -8.5747  12.6201   0.2318   3.3341   6.3987   0.1831 -10.2640  -1.7201
#>    2.9672  -3.5561   7.0965  -9.5050  -0.4171  15.4999 -10.8932  10.4020
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
