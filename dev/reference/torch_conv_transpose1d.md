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
#> Columns 1 to 8   2.3150   6.9998   4.2179   7.5874   9.5413  -3.2639   0.2237  -3.9675
#>   -3.4281   4.2945   2.3772  -3.0493   6.6930   3.6555  15.2015  -5.0496
#>   -2.5798  -1.8901   0.7332   2.2317  -1.1439  -8.7580  -8.4128  -1.4935
#>   -9.3317   9.3308   4.6051  -2.1421  -1.3024   6.6571   7.1674  10.9795
#>   -4.0748  -3.4434  -2.2586  -2.1764   7.3453  -0.1638 -10.2554  -4.2631
#>   -5.3346  -6.0611   3.7701 -12.7099  -0.0225 -10.3141  -6.7695   4.5676
#>    3.2393 -13.0206  10.8507 -12.8139   1.0652  18.8263  10.9322  -2.6205
#>    2.0288  -0.9535  10.3608  -1.5630  11.6139  -3.3921  -6.3445  -8.9061
#>    6.7548  -6.6320   6.1460  -4.5186  -6.6201  11.1091  -1.0883  -3.1876
#>    1.9290  -5.4290  11.1507   0.8897   9.5305 -12.6960  -8.3082  -4.0540
#>   -5.2724  -4.3035   4.8461   6.2092  -9.2015   6.5307  -1.5157 -10.8197
#>    0.1600   5.3395 -14.7173   4.5308  -0.0711   0.0548  14.3201  14.9327
#>   -4.2823  13.5703  -2.5692   1.2683   4.5272  -2.0535   5.2202   0.8730
#>   -3.8990   0.5234   9.4557  -1.0519   4.8540 -13.9025  -5.9222 -14.7649
#>   -5.8815  -4.7005   1.1952  -3.1453   9.1444 -22.2667 -11.7976   9.4945
#>   -3.1575  -4.9566   4.9478  -0.3777   5.7798  23.1028   1.2711  -4.8030
#>   -3.1167  -0.8286   2.7529  -5.1547  10.7612  10.5478   4.7369   4.0405
#>    3.9670  -1.3164  -0.8621  -4.6411   0.1250  13.9993   3.1432  -9.6440
#>   -0.1767  -4.1466   4.5462 -15.0669   4.1676 -25.5780  -5.5599   3.4772
#>    3.9819   6.6481   1.2190   7.0086  19.4851  -1.8290   7.5133  -3.3177
#>    5.9859  -1.8475   2.0505  -1.0050  -8.8080 -20.3313 -24.9048  -0.1986
#>    4.3886  -0.2749   7.5869  -7.4273   0.3747  -1.3233   9.3953  -7.7634
#>   -2.0268  -0.9310  -5.0620  -6.0596  -0.1675   0.0376   1.6211  -9.2135
#>   -5.6291  -2.4439  -2.7274   0.6691  13.4023   5.5755   2.0530  15.8783
#>   -4.7120  -7.6691  -2.0504   1.7751  10.8274  -4.0736   4.0321   3.6285
#>    1.5698  11.5862   5.9976  -8.4264  -1.5772 -13.5056  15.5862  14.8669
#>    0.8322   5.5594   8.4007  -2.6705  -6.4994  -3.7186 -13.4639  13.1956
#>   -0.6342  -6.2824  -3.5684  -9.1931  -3.9517   1.0115 -10.9032 -15.0171
#>   -3.0245  -1.6807  -6.7257   8.0153  -3.8274  -5.6141  17.1022   4.5464
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
