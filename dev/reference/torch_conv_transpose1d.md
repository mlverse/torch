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
#> Columns 1 to 8  -5.9510   2.2601  10.6875  -9.6246  -8.1026  17.7510 -10.0588   4.9733
#>    4.4084   0.5324   4.8212  22.3780  -3.9623   2.4830  13.3726  -6.2502
#>    5.2169  -7.8205 -12.5406   3.5857  -1.5961 -14.0774   6.3071  -4.3301
#>   -1.4720   8.0641   2.8778 -17.8999 -23.7373  -4.1679   1.7718  -6.4444
#>    0.5135  -2.1613   0.5614 -15.9751   3.3848   0.0285  -0.6047   5.2679
#>   -2.1346   3.9496  -0.4825  -5.4150  -9.0808 -13.1936   5.4984  15.1986
#>   -3.3018  -8.6113   8.8282  18.1638 -12.6872   6.2755  15.2841 -12.6958
#>    0.0795  -0.1217  -3.8805   0.4695   1.2282 -25.2745 -10.8463   9.4069
#>    1.3098   0.4432  -4.3384  -3.0038  -2.8094 -10.7319  15.1356  15.5563
#>    0.9991   3.8027   3.0473   6.3982   3.8208   6.1230   0.4522  -9.6491
#>   -3.8495   3.8266  15.9129  -3.8135 -15.3313  -1.6604  14.1415   3.6489
#>   -5.1849  -0.4610  -2.8828  -4.4994   7.4867   2.9606  18.7133  -5.7503
#>    0.7308  -5.9672   3.5653   4.0418  -0.9623 -14.8792  11.6484   7.4718
#>    3.6400  10.9120  -4.3225  -7.0885   1.6378  -6.0676  -5.3819 -14.1623
#>   -2.1513  -0.9725  -1.1957  -3.8751   0.2670   3.7524  -7.2674  -6.0551
#>    4.4773   6.9159  -0.5492  -1.0841   5.3893   3.5459   1.2315   2.2210
#>    1.6479   1.0189  -1.9010   4.1177   4.0833   4.0195 -10.5618 -10.4742
#>    0.2836  -1.0571 -10.1674  12.4240   3.6653   4.6681  -1.0958 -10.6581
#>    0.5346 -10.5412  -0.9744  -2.9970  -9.2689  -6.2709  10.2987   1.9973
#>   -5.2376 -12.0916  -4.4039  -9.4046   4.6948   5.6133  -4.7291  12.5430
#>   -3.7130  -3.3873   8.7870   4.4547  -1.0280  10.3279   2.3116   3.0244
#>   -2.1048   2.0876   8.3875   2.5592   8.6263  -6.4643  -1.0188  14.6580
#>   -2.1069  -9.4507  -7.8471  12.2933   4.3206 -20.1823   9.1717  -2.9210
#>   -2.4127  -2.1306  -4.7789  -4.1036   7.6114   4.2031  19.8405 -10.3448
#>    6.0702  -0.7120 -10.1262  -4.7802   7.7298  -6.5474  -0.0098 -11.0387
#>    5.0165   0.6820   4.5833   5.2732  -6.3472   2.6507   8.5305   8.1745
#>   -2.8920  -4.5465   4.5645  -6.6772   6.2120  -3.5872  -6.5527   3.6259
#>   -4.1938  -8.9863   1.6746 -15.9674  -8.8941  -6.1819  19.8601   8.7819
#>   -0.5732 -13.3689  -9.3348   0.1852  -3.4621  -2.2989  -4.0415  11.5087
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
