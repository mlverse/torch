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
#> Columns 1 to 8   1.2511   0.3253   8.6829  -1.3790  13.4580  -4.9737  15.2473   5.7682
#>    1.5887  -1.7342   1.1091  28.3902  -5.2531  -3.1725   7.0762   8.2121
#>    0.0637   3.2296   4.7754  -9.2170  -0.7779  12.2453  -1.1398  -2.2542
#>   -1.6334   6.8457   0.5296  12.9503  -2.4227  -5.4251   8.7930  12.3523
#>   -2.0204   5.7946   0.3221  -5.5496  -5.6198 -10.1741  -0.7897 -10.7030
#>   -0.4076   3.5288  -0.6784   0.9641  15.9257  -8.9530 -18.0876   5.5044
#>   -1.7739   3.8818   2.7057   7.6671   3.7806   4.0333   1.4294  -8.9760
#>   -7.8913   6.4429 -10.4098   2.9561   5.2543   4.5072  -3.2594  -0.9819
#>   -1.5459   4.8181  -3.6272   2.8905   5.3483  -2.3294 -14.0738  -9.2660
#>   -3.7375  -0.0521  -0.9497   6.3673  10.6564   8.0995   1.3083   3.6087
#>   -8.7452  -1.1966  -8.3769  -5.6688  11.1706  -2.7506 -12.6213  -1.4492
#>   -5.0393   8.9512   2.6441   7.1457   7.8406   4.9271 -16.2055 -21.6708
#>    0.9823  10.8164   4.7327  -4.2121  15.5709  -3.1977  -1.5063  11.3335
#>   -3.2552  -9.5791   2.7029   2.4154  12.0624  12.3669 -23.8839  -2.6049
#>   -6.2659  -0.5153   7.1262   6.0866  -3.1435  11.8642  15.8319 -13.8535
#>    1.0665   0.2758 -18.3198   8.1909   9.6532  -6.6759  -2.2767   4.9678
#>    1.7285  -4.5990   2.4612   0.2557  13.2194   2.9010 -11.2171   6.9305
#>    0.6903   9.6816  -5.0625  -0.6473  -4.3775   1.9750  -1.0097  11.7786
#>   -2.0818  -0.2206   0.3345  17.6761  -0.4789   8.7220  -5.2170  11.3634
#>    4.0079   3.5708   6.2886   3.4349  -2.0538  -2.3729  -8.9695  14.4131
#>    2.7107   7.6937   1.4513   4.3526   3.3549 -12.4979   2.3565   9.3108
#>  -10.3398  -4.3123  -1.9026  -5.1406  -4.5321  -2.6706 -11.1691   2.0055
#>   -5.4039   4.3970   7.6723 -15.6494  12.9139  -8.0399 -11.5508   6.8463
#>   -8.1732   6.4011  -0.7529   4.6433  11.5630  -4.7453   1.4625  -6.9766
#>    4.2582   8.7729  12.7373   0.4468   8.4711 -16.9924  -5.2178   4.6909
#>    7.2536  10.7802  -1.8106   8.5756   1.5730 -21.5303  11.8807  15.3895
#>    1.3311  -3.9636   2.3002  -5.7646   1.4438   6.8205   2.5103  -6.6679
#>    3.2696   0.3498   1.9883   6.3434  10.8871 -16.7037 -23.6414  19.6193
#>   -0.1856   0.3222  -3.3815  -1.9792  22.1316  -4.9065  -4.3529  11.8987
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
