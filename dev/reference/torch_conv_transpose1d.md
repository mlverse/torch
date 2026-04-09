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
#> Columns 1 to 8  -3.5403   8.2782  -6.5998  -0.9737  11.0543   0.2146 -18.0237 -15.9832
#>   -1.4967   2.6002  -6.6018  11.3631 -10.8250  -6.6560  -5.1963   1.1531
#>    4.0087  -0.7025 -13.2249  12.4821  18.3296  -3.7623   0.2899  -6.6122
#>   -0.4457  -2.6893   7.9054  -4.3674  11.9959   0.1793  -5.2079 -10.7675
#>    4.9912   3.5774   4.0000   1.5345  -4.4732  -6.2428 -11.1388   3.4393
#>    1.3362  -4.6627   3.8620   8.2098  -9.9489  -9.2937   7.9616  -5.8765
#>    2.7788   6.2419  -1.6917 -18.8150   8.2479   7.3581  14.6782  -6.5055
#>    2.8007  -4.0240  -3.7363  -1.4524  -3.1734   5.0910   2.1617  -6.5930
#>   -0.2328   2.8577 -11.0800  -1.0976  -6.6246   3.2947  12.2601  11.8485
#>    5.8971   7.4394  -5.1204  -6.8921   1.0678   3.5467  -9.8372  -7.6876
#>    1.7524 -12.9493   4.6773  -0.8379  -7.5264  12.4127   1.8725  -6.3303
#>   -0.6848  -8.8500  16.7978   0.2981  -3.1333   1.3740   0.1943  -8.6340
#>   -3.3831  -2.3641   7.4194  -8.8749  -8.0347   4.8680  18.0501   6.1477
#>    4.7156  -3.9833  -1.7712  -5.2413  -8.9860  -2.7496  17.0318   3.7236
#>   -2.5269   0.9592   1.6641 -11.9757  -4.7850  15.1219  13.3487 -20.3273
#>    4.5161  -1.1007   5.0967   3.7292  -0.7972  -3.6363   3.6137  14.2350
#>    4.7958  -0.7399   1.2381 -15.0217   7.5394   6.4919   1.6409 -14.2516
#>   -4.5248   6.4812  18.1430   7.8722 -11.2238   1.4770  16.6912  -0.0916
#>    7.2717   9.0417   0.1722   8.7371  -3.4235 -13.4886  -3.5837  -3.3062
#>    0.5115  -1.9147   2.1646  -1.8953  -0.4468   0.6267  -4.8999  11.3407
#>    3.5042   1.7518  -2.9572  -8.6171 -13.6040  -6.0316  15.5549  10.7793
#>    7.7033  -6.0588   4.5981   2.7140   5.7656  -0.3314  -4.1231   1.6052
#>   -2.4064  15.7431  -6.5967 -11.4407   4.7588 -16.9961  -8.5461  -7.2416
#>    0.0956   5.3108   2.6534  -3.2468  12.0762   4.9486  -4.2742  -1.4485
#>   -4.9713  -2.6827  16.6098  13.6031   9.7575 -15.4154  -2.0311   8.6369
#>   -5.6988  -3.4443   1.9725  -7.2914  -5.7147  -3.9699  -6.4256  -3.7991
#>    3.7283  15.4547  -0.5758  -7.0497  12.2722 -10.5294 -14.3070  -7.2605
#>   -0.1817  -0.5971  11.8810  -6.2478  -4.3936  -5.3997 -10.3462   1.7228
#>   -1.8181  -4.0336   7.0413  -5.9036   5.5442   3.1835   4.6787  -7.1645
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
