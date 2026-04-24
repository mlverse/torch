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
#> Columns 1 to 8  -0.6567   5.1731  -2.4840  -3.6006  -2.6138   3.7193  -4.8577  -2.8522
#>    0.7610   0.8470  -0.9628 -11.9505 -11.6635  -8.3517   0.9169  -1.4266
#>    7.5859 -10.7526  -4.4719  -1.7967   3.8492  -0.7062  -5.7016   5.3896
#>    2.1955   0.4356   0.6399   3.0300   1.0657   1.7915   7.7857   3.2461
#>   -1.7898   9.2573   5.7044  -2.4225   5.2876   5.0400 -14.7944  -7.8701
#>   -0.0500  -4.0542  -4.1397 -10.9885   4.1048   8.2505   6.0036   1.5990
#>    0.0540   3.3792   4.0349  -6.8837 -15.6046   7.1868   6.7291  -8.1037
#>   -6.4671   6.8022   2.0420   4.6744   7.5315  -4.6805  -4.8498   0.0775
#>   -3.5511   7.0228   5.1375   7.4848  -4.1177  -5.5560  -6.5063  11.0122
#>   -2.5024  -4.5170   3.2997  -7.0683   2.6859  17.8923   1.8597  -8.3640
#>   -3.8737   1.0354  -0.8053   6.0173  17.2255  -5.1648  -3.1820  -2.7644
#>   -1.5972  -0.8587   6.5897   6.8554   0.4129  -0.3224   3.0919 -16.5373
#>   -5.3211  15.6963   1.8157  -1.3921  -3.0824   7.8074   6.1175  -6.1584
#>    2.1661  -1.2721   3.0883  -8.0993  -8.9271   4.5633  22.6133  13.8233
#>    0.5104  -0.4244   5.0892  -2.9059  -9.0799  -0.9323  12.6918  -9.3442
#>    2.0524  -0.5204   1.7978  -0.5857  -5.3213  12.3951   4.4385  -1.4437
#>   -1.9087  -2.7954  -4.5647   7.3633   6.9468 -10.9012   3.0931   0.5927
#>    1.4050  -5.7448  -1.2818  -5.5928  -6.8140   3.8528   2.2242   0.0309
#>   -0.5975  10.1130   0.0317   5.2148  -6.6029   1.7749  -2.4927  10.3536
#>   -2.7026  -1.1591  -1.3126  14.2992 -11.8634  -8.0304  10.6939  13.2038
#>    8.2775 -13.0577  -8.2512 -11.9947  -1.7983   4.6858   6.1283   0.2699
#>   -1.8478  -9.5185 -16.5244  -4.1819  -6.8655  11.3154  -4.6572 -14.0826
#>    3.6809  -4.4048  11.1268   5.6131  -2.3397  -1.9427  -0.0871 -12.2903
#>   -0.2086   8.7512  -0.3035  -9.8005 -10.4818   0.7896  10.4796  -3.4214
#>    0.6183  -0.3236   0.0826   9.2016  13.0706 -16.8533  -4.0838  -0.4316
#>   -1.8351  -1.5387 -16.0489   1.5318  -0.0984   4.0648  -1.7815   2.0348
#>    0.1799  -4.2863   2.9960   4.4871   6.3133  -2.0220  -5.7422 -18.0895
#>    1.7821   2.4130  -3.9585 -11.7649 -17.1067  -5.4672  11.0223  -5.7386
#>    1.6452  -3.5485   2.5203 -20.1188 -23.7361  18.1225   3.0919 -13.4369
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
