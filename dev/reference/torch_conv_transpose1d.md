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
#> Columns 1 to 8  -2.1180 -12.8608  10.5901   7.1954   4.0051  -6.1885  -0.9865  -0.9526
#>   -1.9659  -5.5357   0.1925  -6.9698  -6.8565   6.6638   3.3329  10.6434
#>   -6.2110   7.1696   8.4380  -6.9812   4.2428  10.1887  -4.0666   2.1332
#>   -1.6403   7.3367   1.2406 -15.3061  -1.2595   2.8606  -1.8775 -18.9588
#>   -0.6085   1.1879  -5.5928  -5.1562  -3.2911   6.5988   0.9228  10.0835
#>    0.1358  -6.4956  -5.8073  -9.3107  -4.1694  16.5018   4.3857   9.1727
#>    1.9068   0.9796 -15.3770   9.1072 -12.5134  -5.3092  -3.4988   3.6605
#>   -2.2791  -0.1773   5.5782  -6.1453   9.2261  -0.7157 -13.7767 -13.3005
#>   -0.1373  -3.6270   8.6299   4.9486 -18.2606 -19.7664   9.4594  -4.8116
#>   -1.5563   1.3953  -1.6125   6.6898  -9.3627 -12.2664   2.5158  -2.9668
#>   -0.5127  -5.1275   5.7992  12.0736  -4.5503   8.4911 -20.3894   9.4812
#>    3.4244  -9.6382  -9.2426   5.1224   5.6040  -7.1818 -12.5521 -11.7763
#>    1.0187  -3.3013   4.0468  -1.0424  -1.5664  -5.5504   4.5666   9.0323
#>   -4.1739  -6.5108   2.3524  -6.2890   8.9013  -1.5317   8.0172  -6.8043
#>    1.8178   4.4816  -5.0223 -10.5635  -5.5800   0.0083   2.1308  15.3820
#>   -4.2838  -2.9400   5.5835 -11.4477  -3.0326   7.7952   1.5705  -4.0618
#>   -1.2174  -5.1014   5.7254  -6.4669   6.5089   7.2846   4.2513  -7.1648
#>    0.2176  -4.3537  -4.5182   3.4534   1.8881  15.8302   3.8629 -20.7709
#>    2.0944   2.9151   6.1299 -10.5836  -1.3595  -2.4203   2.5028  -2.1139
#>   -4.6294  12.5188  14.3302  -4.4156   5.9703   9.1032   0.3982  -8.6189
#>   -0.9984  -4.6278 -12.1601   6.1251   9.7166   5.4759  -3.3904   7.6516
#>   -3.7448   0.5858  -8.2767   9.2254   3.4783  -2.1752   7.6639   0.9322
#>    1.1368   5.3190  -2.7791  11.7555 -16.0262  10.6087  12.3634 -13.5309
#>    2.7001  -0.6243  -2.9940  -0.5761  -4.5589   1.8149 -13.4692   0.3084
#>    1.8183   4.1785 -15.5828  10.9351  -1.7951   7.4944 -10.7690  -2.3410
#>    0.5447  -1.1609  -1.5398  -3.6979   9.1894  -6.5104  11.2368 -16.2406
#>   -1.7727  -1.7747  -3.3757  -3.4893  -2.5484 -10.6118  -1.9552   5.9611
#>    2.7236   7.9901   0.1854   7.4273 -10.3968  -5.9873  11.4561  -5.2207
#>   -1.6709   5.0664   6.4809 -14.7031  -7.7112   6.7557  -3.3428  16.9320
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
