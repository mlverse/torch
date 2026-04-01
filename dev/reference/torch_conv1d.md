# Conv1d

Conv1d

## Usage

``` r
torch_conv1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv1d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See
[`nn_conv1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

filters = torch_randn(c(33, 16, 3))
inputs = torch_randn(c(20, 16, 50))
nnf_conv1d(inputs, filters)
}
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 8  -0.2435   4.9549   1.4751  -2.4271   5.5999   2.3987  -4.8038  -5.0209
#>    1.1274  -8.7641   3.1641  19.5059   3.8117   5.0231   2.4428  -0.1971
#>    7.7969  -8.8017   0.1148  -4.1990  -3.5821   1.9739   6.7988  -5.7363
#>    2.4071  -4.6479 -12.0220   4.2109   1.6231  12.4321   7.7793  -2.3321
#>   -7.2780  -2.4390  -5.7303   0.3223   1.8976   0.9882  -6.1879  -0.6613
#>  -10.8257   4.8159  -4.4343  17.8109   6.1121   4.5977  -6.2052  -9.0269
#>    2.0871  -7.3572   4.9592  -4.1701   8.8454  -5.0025  -2.1180  -1.0433
#>    6.3052  -6.3159  10.3824   0.3905   0.0628  -1.3727  -0.3737  -7.5340
#>    5.2480  -7.9646  -8.4808   7.4745  -1.9571  -5.0947  -0.2121  -4.4733
#>    0.7735  -6.1267  -1.7552   4.9839  -2.6537   2.9480  -1.2041 -12.6480
#>    6.3904  -7.1937  -9.1130 -10.0261  -2.0717  -3.7526   2.7276  10.3803
#>   -0.2572   3.1371  -4.4939   8.1439   4.2293   3.4647   7.3538 -12.4630
#>   -0.2330   4.3480  -4.2327   4.9785  15.6584   1.6533  -3.6364  -1.7818
#>    1.1895   2.6088   2.4149  10.4435  -1.3216   2.5942   5.0207   0.3576
#>   -9.1561  -1.1570  -5.0606  -8.2870  -3.9443   4.1766   1.7729  -5.6824
#>   -2.4187   2.5187   1.7276  -4.5173   0.5566   4.7222   9.2208   5.7605
#>    0.5930  -6.1667   1.6037   5.3871  15.7583   4.5987  -8.2756  -0.6673
#>   -5.4071  -6.4596   0.3667   4.2688   4.7255   8.5007   4.0283  12.8222
#>   -8.3508   3.5379   7.9541   3.5293   5.7806   5.8766  -2.7408   9.2195
#>    2.6763  -0.2473  -4.8906  -0.8087  -2.0890   1.3140  -6.0044 -10.9615
#>   -2.6424  -5.7776   0.7088  12.9554  -3.6007   0.6312 -10.1420  -4.5089
#>  -10.2162  -1.1324   1.5651  -3.1705  -5.4946   8.6230   1.2463   0.5543
#>   -7.0461   7.6220   9.6023  -1.6021  -1.8904  -5.5029  -7.8224  -1.2023
#>   -6.5820   2.7395   6.1504  18.8960  -5.8324  -1.0963  -2.4617  -6.9511
#>    8.3072  -6.0815  -9.1080  -8.0883 -11.1876   4.1092  15.8426  -5.9041
#>   -9.0001  -2.5809   5.4164   5.4809  -0.5554  12.4176   4.6600  -5.9774
#>   -0.0091   5.8424   9.0466  -3.6365  -1.9796   0.6828   1.2214  -6.1163
#>   -3.6968  -3.4113   2.9686  -2.4740  -2.6107   4.5555  -2.2562   0.0886
#>    9.2047   7.2096  14.8137  -8.2938   1.8003  -6.5032  -1.6123   3.9234
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
