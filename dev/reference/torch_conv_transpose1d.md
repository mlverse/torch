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
#>  Columns 1 to 8   3.4682  -7.5091  10.7904  -5.3274  -7.7957  -2.0884  -1.2436  -7.2700
#>    2.2721   3.3209  17.1239   4.1401  -4.5217 -23.5295   7.8634  11.6766
#>    2.5342  -3.4695  -7.7762  -9.1868 -13.2396  -4.2287   6.4602 -17.0869
#>    3.2940  -2.8228  -6.3203  -5.1571   1.1615 -11.3901   2.7903 -11.8346
#>    0.7222  -3.8218   9.6769  -0.3276 -20.6360 -15.3072  -1.3972  -3.6219
#>   -2.8883 -12.3674  11.2135  -1.5834  11.4257  11.2601  -3.4387  24.7139
#>    3.7384   1.7058  -8.4425   6.7294 -15.3528  -3.2995   1.9941 -26.5233
#>   -5.4408   8.2324   1.1286   7.8826  -5.0652  -0.4733  16.6822   1.8841
#>    1.8711  -4.1514 -10.3025   9.2307   5.7859  -4.9900  -0.4955  -5.7956
#>   -1.2803  -4.1332  -2.0813 -11.5473  -8.7006   8.4472   1.0196  -2.2042
#>   -3.4063  -5.3163   2.8829  13.2455   1.5990   1.6596  -0.2591   7.4894
#>   -1.2621   3.3599  -4.4781   8.6007  -2.1695   3.5746  -5.6352  -6.5015
#>    1.1400   3.2839   0.4716   6.0707  -2.9609  -0.9520   4.9882   9.1532
#>   -1.9646  -2.8896 -24.2170   4.5497   0.6360   8.7806  -5.0707   7.4367
#>   -0.5253   8.0793  13.2462  11.1343  15.1820   4.5573 -10.4769   6.1908
#>    0.9684  -3.4403  -5.6634   6.6068   8.7120 -12.6715  -1.9258 -10.7979
#>   -3.3551  -2.4853  -3.9025  -7.8989   1.9485 -13.3532   2.4297   1.0567
#>   -0.6257  -3.6717  -2.5786  -2.6322  -9.8624   6.4392  -0.8857  -8.5944
#>    4.9295   6.0048  -1.9097  -5.1213 -11.9266  13.7244  -6.0483   3.2652
#>    1.8939  -0.1110  -9.0516   2.2886   0.7803   8.5559   8.4637 -14.9293
#>   -1.2999  -2.6955   0.3532   0.7665 -17.8767  -0.8644  -0.9995  -6.9745
#>   -3.8555  -6.4528  -4.5371  -5.8658  -8.5097 -12.6028   1.5277  -1.0782
#>   -0.4718  -6.0443  -9.8931   0.4361 -17.7476   0.2684  -6.4389  -8.3894
#>    3.2455   9.7199  12.4113   3.9546  -7.1720   8.1555  15.5018   6.8243
#>    1.2632   8.6182   4.1468  -0.2252 -15.7796  13.9302   6.9697  -2.2329
#>   -1.3856  -7.8927  -8.6110  15.6309   2.9814  -4.3117  -4.3261  -7.9710
#>    5.7434   5.0810   1.9402 -10.9661   6.6102  17.1618  -3.0941  -9.7394
#>    5.2433  -9.5096  -5.6090   0.7588   8.9831  -2.6470  -4.5802   4.2594
#>   -2.1771  -1.1650  -5.9117   4.4227  -2.5473  -6.4462  -7.1155   2.9608
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
