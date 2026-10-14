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
#> Columns 1 to 8  -7.5040  -2.8905   6.5723  -0.5852  15.2629 -16.2141   3.0108   4.4423
#>   -0.7022   6.7617   7.1964   5.1145  -0.5252  -8.6331  -2.5060   4.6448
#>   -2.1380   0.0238  -2.4465   4.3647   3.5462 -12.5423   2.5001 -14.4842
#>    5.2350   5.0572   1.6644  12.1524  -6.8995  -4.6359   4.7265   2.8056
#>    3.2432   3.7233   1.2434   9.4383   1.4858  -3.1444   2.0230   3.9058
#>    5.1441  -8.2422  -0.9776   6.3577  -4.2327   8.1443   0.6156  -2.5410
#>    2.3329   9.2731   6.6933  -3.6366  -3.6915   3.0052  -4.1834  -9.2725
#>   -1.0588  -2.4982  18.1290   3.9629  -4.2460  -7.9756  -6.8528 -15.5653
#>   -3.5805   2.6309   7.6766 -10.4040   4.7042  -5.3868  -3.1330  10.4609
#>   -3.9993  -4.4949   5.7321  16.5895   6.7247  -1.0609   1.1756  -2.6020
#>   -2.7004  -5.2356   1.4754   1.0836   4.0325 -14.0815 -15.9906 -17.6046
#>   -9.9447   4.0622   5.2176  -9.4191  -6.0981   4.1301  -2.0876   5.5261
#>    6.5483  -0.7834  -3.0366  -6.5747  -0.6392  -2.6231  -4.3756   3.1653
#>    0.2569   1.7954   0.8894   0.8226   4.8073  -3.1490  12.8193   5.5610
#>   -2.0420   1.5252  -7.5053   4.3181   8.1242   1.0582   5.5426   0.4682
#>   -5.1609  -7.0285  -5.1530 -11.1707   8.4070  13.5229   0.9485   8.8576
#>   -1.7083  -3.7195  -3.7824  -5.0347  10.1454   1.5903   3.1057   8.1381
#>   -4.6017   3.1069  -1.4950  -2.3251   7.1597   5.9986   2.1567   4.7702
#>    3.9779 -10.0694   0.5448   2.6180  -8.2953   2.3357  -7.3997   3.5808
#>   -0.9193  -9.5666  -0.5110   0.3213   2.0836   6.7193  -8.1633   0.5691
#>   -0.3185   1.7912  -3.4626  -0.2108   8.0090  -9.7861   5.3047  -2.0442
#>   -7.0942  -7.2582  -7.0517  -1.5840  -0.6559   2.4120   6.4105  20.6822
#>    2.5449   0.2715  -1.6526   8.9940   0.2532   9.7449 -13.2053  -2.3981
#>   -0.5487  -9.5574   5.4300  -0.1024   3.5689  -1.4347   4.0063 -10.2459
#>    1.2595  -0.9064  -0.8032   1.9369  -6.0861   6.5822  -3.2948  13.0676
#>  -10.8245  -4.8742   0.7878   7.0071  -6.0294   6.8766   4.9944   7.5902
#>   -2.9469   4.0873  -1.7715  -5.5315   0.4187  -0.7908   0.3353  -0.1321
#>   -7.2498  10.4306   7.5697  -3.6898   1.3146  -4.2968   6.0470   6.4186
#>   -1.9201  -2.6767   0.8164   6.4417   1.8992  15.0835 -18.4496   5.8966
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
