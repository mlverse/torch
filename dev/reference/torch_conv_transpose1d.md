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
#> Columns 1 to 8   1.3515  -3.7266  -5.7831   5.2091  22.9494   2.6761  13.9318   0.0793
#>    2.2734   0.3697  -8.1579   5.7703   7.6522   4.6441  -5.4833   3.3806
#>    1.1300  -6.8536 -16.8109   4.3280 -14.6081  -2.6906  11.0645  -2.4599
#>    5.4883   1.7539  -0.2276   2.6520  -3.3500  -4.6394  -5.7198   1.1704
#>    8.2440  -3.6143  -1.5194  10.0782   7.1189   0.0718   6.3038  -4.4263
#>   -3.3366  -2.1637  -2.1145  -6.9715 -12.2195  -2.4531   1.7087  -2.5433
#>   -0.0355  -1.3204   4.8101   3.9518  -9.8082  -7.2312   5.6285  -1.0455
#>   -6.8630  -1.5795  10.8458   0.2439  -9.4969  -2.3605  -1.7177   7.4000
#>    0.2970   4.6223   5.5616  -6.5201  -1.7804   2.1567   1.8722   0.2893
#>    6.6486   3.8064  -7.8610   8.3915 -14.0697 -10.9246   2.7229  -1.9047
#>    2.6356  -0.7600   3.7509   3.1350  -5.5107 -14.2579  -1.3310  -5.1442
#>    3.3240  -4.8125   2.3386  -6.4594  19.0844  -6.6816  -5.0500  -5.4738
#>   -4.7905  -6.3786  -8.3172   9.7634   5.5911  -6.3730  -2.1309  13.5491
#>    5.9584   1.6386  -6.7441   0.8135   2.4432   7.2300  11.0087  -0.0335
#>    4.2938   3.9595   2.6328   6.4494  -3.7763   5.0529   0.9032   9.7058
#>    6.4729  -2.8021   8.1628   3.3445 -10.6888  -1.8729   0.2849  -0.0455
#>    7.4913  10.2299  -4.0223   5.8647  -4.4366   1.4361  10.9442   7.1146
#>   -4.1706   4.1322  -4.8263 -12.9839   9.4786   7.4453  -2.6380  -8.7048
#>    5.9006   4.5858  -0.3259  -0.9864   5.1366  -8.0369  -7.6131   8.1354
#>   -5.5418  -3.8645  -6.1177  -7.0348   1.5394 -10.0917  -0.0873   3.7369
#>    0.3507   0.0043  -2.6174  16.6482 -12.8896  -4.4097  12.9957   9.4912
#>   -3.6537   4.7780   5.3523  10.9267   1.3634   6.8516  14.0893   4.3780
#>   -1.4470   2.6287   6.0058 -16.2844   0.3078   6.5824   4.7671  13.2269
#>   -0.2062  -2.8051   4.4498   5.8733  -7.6885  -4.1762   3.1529  -2.8583
#>   -0.4022 -11.9157   2.5334   0.4281  -3.8677 -12.5109   6.2641  -9.1441
#>   -9.3412   5.1619  -9.7548  -2.8024  -2.4688 -10.7437   4.1185   0.4203
#>    2.4094   6.2815 -15.5889  -1.0880  11.0183  -9.5933 -13.2208   9.3028
#>   -4.2008   6.3883   2.6097   5.1819  -3.7136   0.9807   2.4812  -0.6126
#>    1.0843   4.0467 -14.0426  -1.8564   8.2255   0.9956   0.3779   6.6462
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
