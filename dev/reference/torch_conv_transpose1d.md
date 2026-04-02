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
#>  Columns 1 to 8  -0.6086  -5.3984   3.0460   1.5795   8.1689  -2.0949   4.9735 -11.9537
#>    0.7646  -5.6207   1.3455   2.1972  -7.3833   3.2776  -3.7780   5.7150
#>    0.1857  -3.2194   5.0871   0.6027  -7.7928  -7.6758  -9.3372   4.9378
#>    6.9538   4.3235  -6.4601   0.1347  12.9716 -10.1763   1.2438  -3.8941
#>   -0.2928   2.4195  -9.0355  -5.8029  -8.4581  -1.2073  14.8757 -15.1256
#>    1.9494   5.4188   1.6473 -11.1803  -5.3929   2.6907   4.1080   0.5694
#>   -3.1467  -4.4380   1.8379  -1.9065  -4.6024   7.6064   1.2602  -5.8106
#>    0.6771   8.1028   6.0047  -4.9588   0.6935  19.0943  -5.6711   7.2040
#>    3.0611 -11.0071  -2.5915  -2.1639  -2.2352  -3.0870   7.4942   8.7833
#>   -1.2478   3.9203  -3.9278   0.1857  -3.4088  -6.1067   2.1748  -7.7727
#>    7.7284   0.1041   5.0807  -5.7789   7.8064   8.4536  -2.6215  -8.4158
#>    5.8823  -1.5436  -2.7931   5.8758   1.3759  -1.0285   9.8654   4.1975
#>    1.4681  -4.2388   7.4020   3.0652  -4.3466   0.0332   7.4215   1.2934
#>    4.0401  -0.6237   2.7312  11.1170   5.2787  -3.8747  10.4391 -13.7387
#>   -3.1864   1.1509  -6.2827   4.5479 -20.1037  -9.0621 -10.6983  -0.0279
#>    5.0071  -1.3907  -7.4712   6.7979  -6.1826 -11.0483  17.9981  -3.4287
#>    3.9074   6.4029   1.4582  -2.6814  14.4054   6.6202  -8.7990  -2.8581
#>   -1.6439  -0.8478  -6.7924 -10.0717  -3.4239   8.9931  -0.0168  -3.6641
#>   -3.1939  -4.9024  -3.9498  -6.8728   3.0964 -19.2603  -3.7375  -2.9068
#>    1.6070   9.7150 -11.7808   0.5488  14.4172  15.4874  -8.7348   0.7547
#>    0.0169   4.6358   1.1989   0.4780 -11.4055  -9.0567  14.8675  -1.4816
#>   -1.5804  -6.4315   1.0685  -2.2016   2.3052  -6.9056 -14.8853 -11.9597
#>    7.4246  15.0867   6.5080 -16.3203  23.1253  -8.5482   0.8563   7.5739
#>    4.5467   2.7381   2.1034   8.0914  -4.9377  -1.4483   6.9989 -13.3718
#>    0.5504  -1.8361  -0.4468 -14.9031  -6.6285 -10.8148   6.3020   3.3693
#>   -4.3128   0.3006  10.2836   4.5095   3.4408   7.0755 -10.9423  17.4183
#>    3.1567  -1.3361  -8.7176 -12.1856   1.7886   7.4405   0.6553   9.1371
#>    4.5101 -10.7384  12.9618   2.2799  -7.6505 -12.0961  -9.4665   5.9961
#>    3.5357  -0.5192   0.6299  -5.1123  -2.0482  -1.3149   0.9196  -4.8597
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
