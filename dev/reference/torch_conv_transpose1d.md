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
#> Columns 1 to 8   2.0010   3.7410   8.1106   2.4777   4.3910   0.6859 -16.5934  -4.0994
#>   -1.1868   8.1416   4.3880 -18.2028 -20.9787 -10.6197  -5.1256  11.2821
#>    0.5966   2.5209 -12.9898   6.3643   1.3113  -9.5123  12.8562  -9.3159
#>   -6.4593  -1.9244   2.6366   5.9784   0.4071   2.8402  -0.4377  -4.9886
#>   -3.3927  -5.9220   1.0443   1.3718  -0.5235 -13.2804   2.2960  -2.0526
#>    7.3756  -8.3150 -12.5354  -3.8082   0.5757   9.7864  10.8435  11.8217
#>   -4.9659   5.4284   4.6893  -9.3398  -5.6526  -0.2532   8.6818  15.5923
#>   -4.7484   2.1046   3.3596   1.1886   8.4986   7.9172   2.7923  -9.1551
#>    1.5910  -6.0809 -14.0375  -4.0479  -3.2743   2.5569   1.9858  -2.6532
#>   -0.4414   2.8773  -8.5405  -0.4227  -0.8553  -1.7957   7.3818  -3.4081
#>    1.0111  -5.2514  -6.9081   0.2720  -9.0552  -5.5651  11.3908 -22.1932
#>  -10.0992  -8.3312  -2.8615  -2.9842  -2.5852  -4.7430  -6.8395   2.3160
#>   -6.5657   0.5423   9.4216   3.1058  -5.7067   9.6987   9.8041  -5.7779
#>   -2.1698  -6.2747   3.9161 -15.0123 -10.0797   0.9794   5.4993  -9.5656
#>    2.3238  -0.9296   0.0396  -0.2601  14.0965 -14.8070 -19.3695  -0.2975
#>   -7.0044   0.5619   1.9025   4.4687  -5.6014  -5.9378   3.7142   4.7677
#>    1.6857  -5.9445  10.2968  11.0847   6.3543   8.7694 -11.0453  -6.3389
#>    1.5160   1.7696   9.2157  -1.6161   1.9947  -0.0169  -4.1400 -16.1133
#>    0.8500   4.2512   7.7982   0.1014  12.6386   7.0779 -15.0215 -16.0027
#>   -2.3898   2.1779  -1.5599  11.3519   0.7449 -17.9450  -0.1853  -2.5926
#>   -2.2491  -6.2574  -3.8065 -11.1032  -5.8108 -16.6998  12.8318   8.5630
#>   -1.3107   6.1407  -3.2568   6.0671  -2.2089   1.1693 -14.8170 -13.1859
#>    3.3385  -4.0986   1.9958   5.3706 -14.7975  16.1496 -16.9070  -9.3556
#>   -0.8031   1.0573   4.2019  10.3187  -4.9149   7.3894 -24.7521  -8.5777
#>   -3.3297   2.5723  -4.4410 -11.8264  -7.6962 -15.8110  -6.2316  -8.2358
#>   -3.4234   4.9979   7.9311   3.6130   0.5855   6.8136  -0.4992   4.7572
#>    1.9256  11.1319  -0.7929   0.3856  -7.9799   0.1332 -20.3275  -1.0416
#>   -5.9552   1.6807  -8.9447  -2.5863 -21.6206 -27.2659  -9.6135  -4.6761
#>    4.2784   3.6327   6.1658  10.8073  -5.8808  -5.7569  -1.8077  -9.0482
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
