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
#>  Columns 1 to 8  -1.4248  -5.6632  11.0430  -5.3475  -6.6548  -1.5271  -4.8814 -12.9419
#>   -3.3503  -2.4239   1.0880  -5.0703  -8.4194   8.3934   6.5400  -0.8750
#>   -1.7114   3.7298  -3.6323  -3.9650   3.5499   0.6868   5.2129  -9.9294
#>   -1.6995   4.8594   7.1253  -1.4115 -10.4171  26.3942  -8.8289  -3.7051
#>    2.3591   0.3519  -3.8771 -10.1661  -0.9417   5.5868  -2.3326  -1.2329
#>   -5.0888   1.6955   4.4906  -7.6338   9.4692   3.1544 -17.2774  16.5430
#>   -1.6221  -2.1692  -6.3520  -1.7890  18.3999  -5.6683  -7.4949   5.5499
#>    0.4059   0.6522  -3.5746  -5.3336 -14.8745  12.3229 -10.1415   0.2139
#>   -0.2121  -2.4210  -1.6931   6.5095 -16.0340  -4.2039  -4.7596   1.3944
#>   -1.9804   4.0379   0.7267   0.6343   1.2583  -0.4735  -0.3192  -0.8809
#>   -4.5381  -4.9583   0.0689   5.7380   7.7207  -1.3135 -15.2719 -14.0937
#>    5.4060   2.3555  -4.2539   5.9518  -6.3986  -2.8231   2.5797  10.2778
#>    1.3746  -1.4491   2.4405   9.0862   1.0818   5.5047 -10.8771  -4.1142
#>    0.0427   2.5435  -1.8171  -9.1034  -8.0792  -2.7919  -9.2533 -19.9171
#>   -0.6561  -1.6953  -4.8892  -3.0220   5.9075   2.2670   1.6562   2.6210
#>    0.7155   0.1564   1.4102   4.4028  -9.3876  -1.3818  -0.6527  -3.3839
#>    3.3587   9.3107  -1.7575 -15.1360   2.2652  -7.0680   3.3832  10.9794
#>    4.5676  -2.1199  -7.2192  -2.7322  -0.9715   4.7412  12.2378  15.4419
#>   -1.8639  -0.0213  -4.8028   6.6998   1.0532   1.9395  -9.3380  -6.5023
#>    0.5430  -1.0787   8.4531   9.2502   3.3425  -1.7882   5.7716   2.7140
#>    1.4124  -6.8806 -15.9497  -1.4935   1.3574   2.5548   5.4796   2.2719
#>   -0.7196   3.1936   2.7757   6.0850 -10.7650  -3.1759   2.1778 -17.8943
#>    1.0584   4.4825  -4.5378  19.3713  -1.8645   3.8178  -8.6651  -6.8642
#>    0.2066   7.1398   1.8604  -1.0549   3.4104   1.4053  -4.0738   4.8223
#>   -0.3661  -4.4244  -4.3600  -0.3987   3.9335  10.1457  -4.7114  -0.1595
#>   -0.8073   1.7421   3.4297   7.6327 -11.9455  10.3874   1.1136  -9.5167
#>   -7.7151   4.0978  -2.6670   2.2929  -6.6536  17.3481   9.8906   6.6777
#>    5.7169  -3.5871   6.4894   1.8700  -1.3414  -0.1992  -2.1683  -3.7189
#>   -1.3870   3.6108  -0.6304   4.9750  -3.4457 -13.5532   5.9496   4.1739
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
