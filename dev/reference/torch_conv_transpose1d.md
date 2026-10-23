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
#> Columns 1 to 8   2.3622 -11.0865   5.0982   7.0914  -6.9776   4.6008  -0.7091  -1.4767
#>    0.4167   7.6438   2.8798   2.8517  15.8979   7.4262  -7.4284  -3.2201
#>   -0.0685   5.0594  -3.2844  -2.9078   6.1552  10.9116  -8.2554   0.6891
#>   -2.2152   4.7111   6.7588  -1.9489   5.5794  14.4385   0.2557   9.2676
#>    2.5930   9.6920  -2.8575  -3.7426   6.8519  -1.8627   3.4387   1.4989
#>   -2.9428   3.1958   6.4590   0.0416   2.6341 -11.8740   6.6324  15.3310
#>   -6.3329   3.0916   5.6078  10.0178   7.9793  -9.5118   9.9450 -11.8614
#>   -2.9065   2.6895  -0.9906  -6.7705  -5.8976  14.1341  -6.4421  10.9777
#>    3.0865   0.9331  -5.9578 -14.6701   8.3863   6.5246  -5.5472   7.2575
#>    4.8370   3.2918   6.8704 -16.6260   1.7318  -7.8738 -17.5496   5.0176
#>    0.1204  -1.7620   3.5738   3.5133  -0.3063 -10.8398  -0.4725  -6.0371
#>    2.2591  -0.0888   4.7530  -4.4382  -3.9620  -4.9180 -17.6089   0.9270
#>   -2.0229  -0.3662   4.0256  10.5942   4.9843   5.6174   6.8466   4.8947
#>    2.2991   8.4152  -7.5431 -12.8759  -3.7259  -3.3345  -1.2307 -19.2488
#>   -2.7181  -9.7568   8.6674   9.4783  -0.0300  -6.9218  -5.3640   1.4892
#>   -0.6714  -0.1320   4.8903 -10.0988   1.4310  -5.6756   9.1532  -1.9363
#>   -0.4734   5.1560   6.5892  -7.1935 -23.8188  -1.8149   5.1088  -5.9749
#>   -0.8262   5.6828  -2.7493  -0.4411   7.7113   1.7611  12.9766 -15.9655
#>   -3.3164  -0.0782   8.3767   9.3844  -1.2341  -1.9674   1.0143 -27.0258
#>   -2.9222   1.2719   5.8207   7.1819 -14.2610  12.7135  11.0155   6.1601
#>    1.6244   2.9965  -7.9640   2.8873  -9.5148   0.9280  14.6403 -15.6339
#>    7.8588   1.1595   3.8413  -8.0056  -5.4409   1.7115  -0.9179   8.4593
#>   -2.3950  -0.4534  -6.3811   2.8042  17.7894 -10.2416   7.5608 -33.7840
#>    0.2593  -6.9646   8.2787   4.3935  -6.5288  26.5430  -8.6188  -2.7119
#>    7.8746  -8.5545  -2.7821 -14.6827 -22.1029  12.2684   1.7908  -2.7513
#>   -1.5585  -9.1078 -10.1683   5.1084   7.3483  10.6591   5.1118  -1.2598
#>   -3.2121  -3.8140  -0.4748   5.2920  14.9890   3.7167  -3.5956 -11.6063
#>   -3.3806   0.8982   3.5192  -5.2238  -0.4432   1.2078  -3.9221 -12.6272
#>    2.4210  -1.2695  -7.5696   6.0377   3.2869  -0.6039  -5.2048  11.6122
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
