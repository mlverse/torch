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
#> Columns 1 to 8  -8.4757  -5.2331  -7.1994  10.9437   7.1332   9.6988   1.0341   8.5189
#>    5.5901   2.9906  -3.1692   9.3995   0.6756   7.0692   4.2079  -1.5397
#>   -3.4711   5.3300  -0.3306  -1.7344  -3.7849   3.4664  -4.5348   8.3438
#>   -4.5864  -5.4835  -1.9957   1.2784   0.6475 -12.1231  12.8066 -22.9841
#>    0.9271   8.9197 -16.8717   4.2675 -13.6541  -7.9216   4.3053 -17.0078
#>   -1.2955  -3.7605   4.1105   0.7763  -8.9664   6.9513  -7.7743 -16.5729
#>    0.3868  -6.3937   5.4803   4.8015  -4.6357   0.9506  -3.4942   3.3506
#>    1.0124  -3.6538 -13.9106  -3.6534   2.9219   2.4998   1.5600   0.4983
#>    5.5537  -8.6542  11.6733  -3.2544   8.8934  17.1559   3.1923   1.9397
#>    4.8224   8.0555 -13.3758  -1.4662  10.5532  -5.3093  -5.9294   0.9245
#>   -4.6019   2.9233   0.6237   5.2530  -1.7311   4.0498   2.1764  -5.9242
#>    1.4611  -0.2459  -3.5695   5.6775  -7.5302 -13.6027 -17.7165   9.6816
#>   -3.4518   8.8830   7.2839   3.5975   2.3327  -5.5492 -15.6715   8.0382
#>    4.3273  10.5990  12.9922   8.6675   9.3774   2.9124 -15.2325  -6.7777
#>   -0.7625  -4.7293   2.7663   2.7352  -0.1418   8.7736  -3.0877   4.6660
#>   -0.1526  -5.8085  -8.6215  -3.3482 -26.5891   9.8853 -12.0520  -6.6071
#>   -1.3146  -5.2519  -1.8264   2.4562  19.4239  -6.5795  -3.4416  -0.0811
#>    1.2637 -12.8999   7.0723  -3.9704   1.4379   6.9033   6.5403   4.3006
#>   -4.8229   1.1157  -1.0002  12.0019 -14.8270  -4.8845  10.2149  -6.9717
#>   -0.0665   5.5275   1.6092  -2.8197   8.2200   3.7955   7.5255  -3.0297
#>   -8.5350   8.9560  -0.0741   5.7079  -9.8435 -10.0875  21.4846 -10.2726
#>    1.9225  -8.8369  -6.0678  -7.2712  -8.5858 -24.0292  -1.0329  -1.7509
#>   -5.1297  -0.9032  11.1780  -2.1889  -5.7458  11.7195  -2.0122 -12.7866
#>   -2.9148   0.0182  -8.5331 -12.8850  -4.3851  -5.6286  -1.2383  14.6055
#>   -3.7003 -11.1436  -3.5778  -1.4423  -2.1017  -1.5923  -7.9301  -9.3790
#>    1.0411  -3.9352  -3.5580   1.9438  14.2719  -4.6232   3.6423   8.5928
#>    1.1030  -1.0107  -0.2156   4.7674  -2.5460 -17.6214  11.5417  -2.6413
#>   -3.7568  -0.6348  17.8715  -0.7989  19.0630  -4.4360  11.4449   3.4699
#>    0.6473  -4.8949   9.1863   2.4729  10.5803  -4.3460 -15.1088  -3.3565
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
