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
#> Columns 1 to 8  -6.3907  -3.7595  -9.0104   5.8599  -4.9950  -3.5565   1.2729   1.5956
#>   -1.1498  -3.2604 -16.3167  10.6873   3.7879   5.7401  28.0881  -0.4331
#>   -5.0576  -5.0809  -8.9416   6.6055  -1.2454  23.2946  12.2779  -0.9850
#>   -2.4228  -3.0265  -0.1283   6.8562   3.1715  11.8043  17.9318   4.5862
#>   -3.9908 -11.8990 -12.6925  -0.3033   3.1646   8.1277  12.2179  13.1520
#>    2.2450  -5.1249   0.0149  -2.2053  17.1133   2.0900   8.0068   9.4393
#>   -4.1846   4.5760   2.2165 -18.6102   3.8967   4.7203 -15.6843   0.3208
#>   -0.2616   6.5843  12.2716  -1.7919  -3.7882  -5.4464  -8.9017  -6.8544
#>   -0.3783   2.3669 -12.3822   5.8039  -0.6161   2.6100 -15.9068   0.2997
#>   -2.8613   0.4382   0.5804   3.6412  -0.9084   9.4303  -1.2505  -0.0242
#>   -0.1220  12.1700   1.0703 -17.9590  -0.2449   3.2113 -25.9979  -4.9112
#>   -4.0947  15.5135   3.0738   5.1773  11.6556  -5.4934 -10.8454  -1.9258
#>    0.8693  -4.2354   4.9418  -2.5105 -13.2404  -2.0126  -9.8094  12.3340
#>   -9.8140  -2.0250 -11.1529   8.3516 -11.7653   4.3477  18.0125 -15.4438
#>   -3.0937   1.2962  -7.8089   1.4869   8.2125   7.1234  -4.2433   0.4918
#>    2.6415  -4.6417   4.7163  12.6341  -4.5779 -10.1411   8.7620 -12.6787
#>    3.5325 -11.5019  -9.4186  -7.2688  -4.4458   1.2945  22.0328   4.6781
#>   -0.3731   8.3752 -10.6491  -5.8691  19.2001   8.6570  12.0377   6.8523
#>   -1.6294  -1.5099  -7.3519  -5.3506   4.1276   9.3811  -5.9425   1.1692
#>   -0.2637  -0.3244  11.6331  -5.3874   0.6607 -20.2489   7.0232  -1.4519
#>    2.7689  -0.9954 -13.5784   0.3548  -6.3151   2.0144   4.3105   8.1076
#>   -3.5157  -2.6494  -3.8046  -0.1859  -4.6671  14.0728   5.4574   3.6968
#>    5.2948  -6.6434  -1.5542  -5.5763  -9.6389  -1.5639 -10.6756  -4.1274
#>    2.3921  -4.7642  13.5500 -12.3427 -10.2764   8.9583   0.1758 -19.8367
#>  -10.3050  -8.3656  -4.1321 -11.6496   8.1022  13.5007   7.6810  -5.5933
#>    3.9399   0.0120 -11.2555   2.4262  -3.0232  -8.2353  -3.1189  -3.9454
#>   -2.4143  -0.6578   5.6960  13.8143  -2.3343  11.4446  15.2453  15.1208
#>   -3.9180   7.7115   3.4182 -12.3847  -8.4908   0.6147  -6.3679 -11.9692
#>   -7.4115  10.2191  -3.3305  10.6980   3.3034   6.2620  -1.1704 -11.6144
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
