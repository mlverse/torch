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
#> Columns 1 to 8  -0.8839   7.4199   6.9849  -1.9771  -1.6528  -5.2992   2.7196  -7.3136
#>   -3.0247   3.1658  -1.3895  -1.6807  -1.0656   0.6687   1.8625  -6.5083
#>   -0.5764   0.5893   1.7120   4.2980  -8.7372   4.3695  -4.9424   7.9993
#>   -1.3611  -2.6829   0.5702 -11.8824   1.4732   2.4898  -4.1953 -11.6686
#>    2.4880  -6.5429   2.4938   1.7040  -3.6765  -4.7572  -6.9481  -3.3664
#>   -6.5491  -1.9939   3.2318  -6.6437  12.9179  -4.9168  -5.3295  10.3109
#>   10.5961   0.2179   8.1377   1.9215  14.2029 -21.5005  -4.5942   3.0856
#>    1.3865   2.1310   4.1196  -2.0660 -21.4487   9.5465  -3.8478  -6.5586
#>    3.5859   2.0515 -15.7429  -4.1959  -0.1109   4.1140 -14.2696  -6.1626
#>   -1.9658   1.9164   3.4309  -1.7765   4.6809  -0.9812   3.3224   0.1650
#>   -5.8573 -10.3856   6.5919   1.9310 -19.1243  -6.8982  -2.8756   8.0337
#>   -0.6099   3.8517   9.3813  -8.5863   7.8560  10.6265   7.8057   4.0488
#>   -1.5483   6.3105  -0.1883  -1.9415 -13.7936   5.8494   1.6293  -0.8021
#>    5.8187 -10.1927   5.8076   3.8785  -0.0744  -4.1468   6.8359  -3.4804
#>   -2.0948   7.6656  -5.4162  -9.3305   2.2623  -5.1085  -7.7937  -4.3163
#>    0.0090  -4.1840   5.8425  -2.5596 -10.2040  -0.9562  -0.8811   1.7344
#>   10.8899   2.1978   9.0910  -3.7732   0.1272  12.3013  -6.8850   5.5561
#>   -0.3462   4.0093  -2.5776 -12.3643  -5.3017  10.7669 -13.8458   2.1350
#>    1.2993  15.4142 -13.4402   1.5302  -4.6635   1.3092   3.6192  -3.4751
#>   -0.2029   3.1984  -5.2490  -1.3085   5.4829  -4.2767  -7.8481   7.2306
#>    3.5599   3.5362  -4.7020  -7.9902  16.0860   1.9505  -5.3560   2.7649
#>    4.2371   9.6685   8.2542 -10.1281   8.5583   6.0198   6.1717   1.6266
#>    3.1055  -4.0757  -1.4327  -5.6659   4.5883  -5.5901  -1.2684  -9.6331
#>   -0.1187   1.1187  -5.7511  -6.7091   1.4499  -7.7284  -2.4733   4.5312
#>    3.9336 -10.5058   5.1947  -2.1530  -4.8139  -1.9117  -5.0715  -4.7298
#>   -4.7910 -11.7059   9.3652   9.6150 -16.6944  -9.6350   0.0040   5.1439
#>   -0.1003  -6.6588 -11.9605   7.3535  -6.7465   4.1778  -6.6165  -0.0861
#>   -2.1316  -4.6513  -9.3278   2.7181   4.6018 -23.9858   7.8218  -2.9870
#>    0.4671   2.3519  12.3875  -1.1452 -14.2277   2.9540  -1.4528   1.2777
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
