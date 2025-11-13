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
#>  Columns 1 to 8  -0.9715   0.1935   3.6679   0.2262  -1.3086   9.5294  -1.3719   1.8028
#>   -0.2098  -4.6914  -1.0726   0.7832   3.0476   3.2304  20.6521   6.7625
#>    5.9126   4.4338  -0.2494   3.9648   1.5349  -6.0623  11.8529   1.6759
#>    0.3465   9.0634   3.6264  -5.9854  -2.3840  15.1224  -3.5955 -16.6646
#>   -0.5045   5.4732   4.2195   2.7121   3.2441  -4.3439   0.8519   3.4799
#>    0.7820  -2.3451   0.3173  11.4507  -3.0853   2.7565   4.7859  -3.3801
#>   -4.6719  -0.7631  -1.6692   6.0063 -13.1665 -10.2928 -15.1271   2.9423
#>   -0.4059 -10.6610   9.5004   2.1035  -6.9089   0.2765   8.3811   3.5917
#>    1.1276  -1.7502   2.1481  -7.1636 -11.7728   7.2791  16.5416   2.8310
#>   -0.5394  -4.3523  -0.2777 -15.3614   6.0148  16.6886  -0.9800  -3.9839
#>   -4.0214  -1.0888  -5.1473  -7.9064  -0.6057  -7.4017  -1.0856   7.7982
#>    1.7081  -0.5021   6.5321  -6.9187  -3.2266   5.5549 -10.9458 -11.1263
#>   -2.2981  13.1231   0.0055  -8.0107  -5.1450 -11.9673 -10.0667   8.7373
#>   -1.1069   3.8615  -4.4248 -15.1568   0.9356   3.3313   0.0116   8.0413
#>   -1.4695  -0.6092  -5.1770  15.1738   8.7522   4.1922  -9.4106   1.1741
#>   -6.0629   8.7937   6.4268  -4.7967  -8.3635  -4.3790   3.1251  -8.2992
#>    0.4043  -5.7392   4.4479  -5.4525  -7.7345  10.2024  12.1160  -9.7324
#>   -0.8337   3.0476   0.6790  -6.2198   5.3503  -5.8210   3.9585   5.9708
#>    0.1166  -2.7986  -9.1447 -16.9341   5.7802  -9.4812  -9.0750  -1.5596
#>   -2.8008   0.1133   0.9827   4.5963  -4.0086  11.6811   0.5389  -3.9402
#>    0.5259 -11.1716  15.1095  -4.4401  -0.2024  -5.6046  -8.2014  -2.6277
#>    1.1310   0.0524 -10.4884 -14.1276  -9.9306   4.4661   0.0211 -11.0807
#>   -1.1008  -5.1902   8.9503  14.6246   2.6388  -1.5064   7.3792   1.3842
#>    0.4932  -4.9811  -1.4897  -1.2464  12.2783   1.1599   2.9348 -14.3203
#>   -2.4711  10.4366   3.7406  -3.4860 -12.8473  -3.8640  -1.3189  -7.8989
#>    0.5219  -4.3738   1.4499   2.0534  -0.0111  -2.9205  -8.6616   5.7518
#>    3.1239  10.2441  -0.0497   7.8444   3.9086   2.1727  -2.0161  12.9236
#>    2.5246   3.0332  -3.2785  -6.5141   0.4604  11.4029   5.9889  -9.7118
#>   -0.7103   1.9541   2.4435   1.3624  -2.6881   2.3625   0.1010  -4.5306
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
