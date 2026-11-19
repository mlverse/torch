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
[`nn_conv_transpose1d()`](https://torch.mlverse.org/docs/reference/nn_conv_transpose1d.md)
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
#> Columns 1 to 8   0.0115   2.6769   6.6779  -3.4520   1.9126 -15.7129 -12.8504   5.8636
#>   -9.3661  -3.3420  -4.1550  -5.1293  -2.3933   1.0171  -5.4971  -1.8121
#>   -6.9226   4.7461   4.1877   3.8487   9.4190 -19.4744  -4.6696  -0.5341
#>    0.7579   1.5276   9.7639   6.4261  23.6624   2.0980  -0.3126   1.4698
#>    8.4070  -6.1104   4.6115   2.7849   5.2166   1.7429 -13.9895  12.1404
#>   -2.1039   4.1730  -2.1910   0.3343  -5.1804   0.2260 -10.5180   2.1168
#>    1.7733   0.0985   0.5627  -7.0534   3.8172  -4.8953  -5.7598  -2.0746
#>    4.2473  -1.6446  -1.0770   6.4947   4.8253   5.9027  -2.0463  -2.1419
#>   -1.4478   3.3495  -5.5154  -1.5061   1.5407 -11.5131   7.9041  -0.8585
#>    6.1983  -6.3562   0.0408   8.7643  -0.9694  -7.1225  -4.8867  -1.7321
#>    3.3914  -0.7091  -5.4916   1.1235   1.2772   2.0371  -2.5430 -17.2453
#>   -0.7926  -3.5323   8.0599   6.9776  -5.5209  12.8872   2.1700  -8.8763
#>    4.6447  -4.1807   4.2591   0.5008   3.9031  12.4542  -2.0164  -1.8584
#>   -0.3585   4.5127  -8.9376  -3.0252   5.7396   9.3369  12.3157  -8.1695
#>    1.8530  -4.2089   7.3217   1.9881  -8.5179  -0.2996  -9.9712  13.0304
#>   -0.6701   1.1099   3.2073   1.6389  -4.0668  12.0696  18.5748 -18.2392
#>   -0.4357  -9.5703   7.3743  -4.8377  -4.5153  -3.7565  -4.9371   3.6285
#>    1.2162  -6.0343   4.8083  -5.5264   9.1500  11.1655  13.2754   5.8280
#>    2.1140  -2.7225  -0.9838 -16.3175   6.1376 -14.3389  13.1727  20.9002
#>   -2.3419  -3.3470  -5.8723  -2.2404  -7.2760  -3.9639   9.0922  -5.8280
#>   -5.8422   4.4701   6.1794  -1.3740  -1.2696 -12.2805  -1.2674   1.7595
#>    4.1660  -0.5725  13.6122   3.2070  -2.3624  -0.6924  -8.8028  18.7132
#>    1.9244  -3.6204   4.0116  -5.4832  -0.7835  -8.8307  -3.2239   7.1359
#>    0.1527  -0.0087   0.9273  -2.4645  -5.3028  -1.1226  -5.2715   7.8650
#>   -5.0728   1.2668  -4.4264  -1.3304  -0.5879   5.6313  -5.5644 -20.5219
#>    6.7063   4.4473  -7.3243   2.2875   0.2436   1.7168   5.3259   5.0679
#>    7.0109   2.8014  -8.7329   4.1378   2.1409  -4.0129 -13.3606   0.2217
#>    0.3639  -0.7140  -2.1145  -3.1509  -4.9054 -11.0462  -6.9094  -1.7982
#>   -2.0466  -1.0813  -1.0848  -0.2713 -12.9525   7.9455   8.0994 -10.9475
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
