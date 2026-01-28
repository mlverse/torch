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
#>  Columns 1 to 8   5.0767   2.0634  11.9704  -4.4633  -0.6019  -4.5192 -13.7075   4.3443
#>    7.0068  -7.5024  23.4955  -4.5064  13.0214   7.2699  -1.4927  20.9828
#>    1.3048   6.2974  -3.5341   7.6606  -4.5504   5.3003  17.3515 -12.6586
#>    0.7782   4.9163   5.3902  10.5427  -7.7201  -1.2464  -3.4614   7.7934
#>    2.5961   7.8843  19.1928  -4.5702  -2.7419 -15.7728  17.1709   8.1221
#>   -5.0604 -10.5205   1.4180   1.0133 -18.8400  -2.3977  16.0431  -7.3529
#>    1.5424  -2.7526  -3.2270   9.3995   7.0393   4.9976 -12.2725  -9.8852
#>    8.1523   0.4219   1.4463  -8.3881  25.5197   3.4332  -2.0351   0.1579
#>    4.8843  10.2710   7.8387   3.4349 -10.3476   9.5595  -5.5292   7.0701
#>   -6.9479  -0.9555  -1.4475   9.7078  -0.9746  -4.8499  -9.9992  -8.4594
#>   -4.5869   2.1149  -8.4259   4.9927  -0.8379   9.7409  -3.5221  -2.5165
#>   -3.9505  -0.0240  -2.0295 -12.1966   3.4032  -2.9257  -3.5769 -28.0066
#>    0.4204   7.0499   6.1888  -7.5819   7.9734  -3.6280   2.3463  -3.6836
#>   -2.0552   0.3116 -11.7169   6.6651  -8.9189 -12.1904 -13.0672 -14.2366
#>    9.1872  -1.0420  18.3550  -3.7982  -8.8093  -1.1661 -23.4142  11.8052
#>    0.2658 -11.7876   2.7296  -0.5973  -2.3369  19.2488 -12.9881  -6.0155
#>   -3.4735  -2.5698  11.8060  -5.7417   6.9788   1.6898  16.4386 -22.1357
#>    0.4347  -2.0556   8.5012  -3.3255  24.0746 -10.2162   2.9703  -9.0124
#>   -0.3642   8.1960   5.4641 -10.4654   3.9126  -4.1103  -4.7018  12.3762
#>    1.5983  -2.4393   0.9712   0.8597   1.0595   0.4525  -1.8033 -10.5086
#>   -7.8982   1.8676  -0.0272  -8.6296 -12.7060 -12.6543   3.8489  12.2981
#>    1.9423   2.7147   8.0451  -6.3985  10.9750   4.6539   0.8394   4.8135
#>    8.8961  12.8840  -4.9524   5.6000  -7.7913  15.1273 -12.0947  -6.8347
#>    2.1124   1.1957  10.2246 -12.0673   4.1363  -1.5045  -8.7909  -5.8016
#>    1.7499   3.1991   2.1332   1.4939  -4.8758   4.3042   2.4123  -0.6856
#>   -0.0089  -4.4177   5.0183 -20.9888   6.6839  -5.5819  -0.5098 -16.5012
#>    4.4835   4.3770   2.4296 -10.2768  -0.1559  15.1246  -0.0062  -7.5283
#>   -3.6305  -2.8039   7.5022   1.2184  -0.9334  -1.5276  -2.8492   4.2072
#>    7.0925   0.2760 -18.1186  -7.4936 -18.8168  13.1113  -5.2062  -6.6417
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
