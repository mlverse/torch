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
#>  Columns 1 to 8   5.0916  -2.6231 -11.3281   5.3929  15.7037  -2.2631   1.3377  -2.2632
#>   -2.0100   2.8914   2.9481  -3.1599  -4.5637   8.9929   8.6733  -3.9605
#>    3.2747   1.9689   0.6574  -7.6592   3.0163  -1.5716   5.7715   8.5916
#>    4.2252  -0.7696   7.4766  -3.0415  -8.7330  16.3733   3.6160   0.5028
#>    4.8881  -1.3180   8.0405  -8.7086 -10.6239  -3.9899  -0.9865  -6.6365
#>   -2.4299   7.4966   8.6876   3.0331  -1.9539  -7.1605   1.6392   0.4805
#>   -5.3839  -3.9044  11.9375  -7.7275  -0.4919 -10.3362   6.6852  -3.0187
#>    8.8150   0.1712  11.0223  -1.8088  -1.9830   1.2930   2.7099   1.4483
#>   -1.4642  -3.0860   9.0680  10.6241  -2.3233  -5.1770  -3.3785  13.8273
#>    4.3665  -6.8432  16.9881 -13.4208 -13.8404  -1.3247 -13.6056  13.7830
#>    5.6133   4.4916   2.4477  -0.3772  -1.0222  11.4892  18.1554 -10.6632
#>    2.9789  -0.8870  -0.4888   7.9223  -1.5761   3.2287   6.5999  -7.5647
#>    0.1244   4.3760  -1.8824  -1.4003   8.8302   1.7674  -6.2679  -3.9617
#>    9.2413   6.2221  -1.7295  -7.5873   5.0104  -6.4292  -7.0307  -6.4690
#>   -1.5819  11.2981 -17.5450  12.1713   1.7235   0.1787   8.4268 -12.5471
#>    2.8143   6.7286  -6.9231 -11.2877   5.0830   6.9985   1.5035  -5.4180
#>   -1.5279   2.4415  16.1471   0.8768  -0.4856   7.9999   3.7229   5.2989
#>   -3.3745   4.3193  -2.2398  -6.9295   5.8856  -0.9494  -3.3119  13.5369
#>   -2.9974   7.5420   1.5660 -10.7188  -0.5142  11.7278  11.4566   8.3845
#>   -3.3038  -3.8452   7.0343  -5.4294   0.4115   9.9547  -1.4734   0.0038
#>    1.7298   1.9967   2.8040   3.3781  -0.3823   5.2449  12.3228   0.0589
#>    4.0723   0.6696   4.9430  11.7074  -6.6531  11.3173   3.3647  -0.0305
#>    0.5341   5.6008  -9.6137  -3.3225  -0.9973  -2.6594  -0.8508  -0.1493
#>  -12.2961   2.6323 -23.4147   9.1581  14.6085   1.8001  -2.5320 -15.2321
#>    2.6106   0.1295  -5.1077  -4.9482 -10.2634  -4.2395  -6.8541  -3.9870
#>   -3.9909  -4.6050  -3.3713   1.1582  -3.3029  -3.8329  14.6776   2.1071
#>   -8.7878  -2.2891  -8.2206  -3.3447  -0.7172  -7.5253 -11.0139  19.6248
#>   -4.1857  -3.2553   2.7025  -7.8796   1.5108   3.4030  -1.4841   4.0272
#>   -6.2315   1.9287  -1.1857   7.3130  -5.0407  -2.3057  -7.8405   8.2763
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
