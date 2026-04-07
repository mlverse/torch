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
#> Columns 1 to 8  -0.7147  -7.1728  -3.3321   5.2181   1.8355  -2.6836   9.9979  -5.0176
#>    6.9403   2.5632 -10.8254  -2.2060  10.2055  -0.5455 -11.4988  -6.9338
#>    0.2954  11.4632  -6.1775   4.5289  17.7886  -6.0858   7.2834   8.0187
#>   -3.3063  -2.2910   1.7158  14.1157   8.0899  12.7194  11.2840  -3.3471
#>   -5.6746 -10.0670  -6.2101   6.8793  -3.0678  -9.3726  -2.7166   1.3372
#>   -1.2867  -9.4827   0.1138  -7.4693  -3.5817 -16.4809 -27.7925   4.3104
#>   -1.6373  -1.0036  -4.1230   9.2638   9.6359   1.1333  -5.8048   1.2911
#>   -0.6693   4.4091  -2.7834  -8.0874   5.9989  -6.0914  -2.6163   0.6527
#>    3.0691   6.8584   1.9172   9.1767   1.6641   2.5988  -9.6593 -14.9796
#>   -2.5834  -2.9667  14.9938  -3.0248   2.6967  11.3570   1.9315   3.2599
#>   -0.2809 -12.0012   3.6703   5.2172   2.6233  -4.0167   4.1559   9.5712
#>    3.9035   3.0434   6.7547   5.9063  -9.7969  15.2498  -3.4199   4.9879
#>   -1.3053  -5.3487  -2.3694  -0.1251   9.6524 -10.6786   2.0151   6.8953
#>   -4.9937   3.6739   4.9106  -0.8027   1.9380   3.5892  -3.0267 -15.2646
#>   -3.4230   0.0169   0.0791 -15.6080 -10.7444  -1.2659  -5.2410  -5.2282
#>    6.0126   3.7840 -11.6009   6.5807  -0.9359  -9.0965  -2.4020 -23.9271
#>   -4.3388   5.3166   2.9808  -5.4898   1.8159   5.8170   3.8157  -0.2163
#>   -1.4679   8.2724  -2.5525  -0.2537  -6.4211  11.2221   3.0100   2.3151
#>   -4.5755   2.3575   7.9972  11.6384 -16.3145  -5.8302   1.9479  -4.2966
#>   -1.1131   2.0494  -7.4046 -22.7497   5.9009  -1.7178   4.9052   3.5188
#>    2.8248   1.8695   0.5953   3.7287  -1.5700   4.4759   5.6193   2.0823
#>   -1.5693   5.9425   4.7580 -15.5193   7.2730   7.7261  -3.9851   3.3804
#>   -0.2973  -2.4228  -4.4394  -8.4008   2.8611   5.1428  -5.3354   1.4963
#>   -8.0987  -5.6517 -10.2286  10.7322 -11.3744   3.3945  -5.9367  -8.5971
#>    0.5363  -2.5606  -2.3051  -2.2536   1.7910 -14.6607  -8.1131   1.9776
#>    1.2293   9.7047   2.4223  -2.4129 -11.3198  10.6550  11.1738   8.7964
#>    6.3134   4.2708   0.7280  -6.0749   9.5208 -11.4657 -15.8609  -1.9987
#>   -5.3695  -2.2178  -1.9484   9.9769  -7.2337  -4.1880 -10.0354   9.0982
#>   -2.3686  -5.6126  -2.3487   0.8779  -8.4520   1.4654   5.4555  -1.9484
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
