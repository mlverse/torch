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
#> Columns 1 to 8  -4.2101   4.0917  -6.8635   2.5394  13.1306 -12.0917  15.5293  -9.3146
#>   -0.6125  -3.9190  -1.7233   0.0439   9.2766  -1.3861  -1.0014  10.5282
#>   -2.2218   2.1428  -1.2412  -3.1159   4.3264   3.4823   0.0585   3.5115
#>  -11.6165  -8.9240  -5.9697  -1.6170   2.4279  -2.5712   0.3239   8.6255
#>   -1.6587   1.6953   9.9150   0.7180  -3.9600  -9.0502  -7.5367  -1.9882
#>   -0.6789  -1.0601 -12.8303  -8.5722   1.0984  -9.4552   7.3956   5.7541
#>   -1.6049  -6.4999   0.1865  -1.3169 -14.9599  11.1795  -0.3132   1.1798
#>   10.2634   6.5392   4.2977   3.8365   2.2558   3.5874   9.0582   0.3828
#>   -9.1054   0.7417   5.8563   6.9601 -11.8269 -12.2426   1.3951   2.1138
#>   -2.1103   2.8798  -6.6172   5.0916   8.0969   9.3954   6.1167   3.2154
#>    7.0053   0.0492 -18.4253   1.9495  -4.8294   1.2057  -8.1582  -7.7129
#>    6.2531  -1.1598  -7.7553   0.0258   4.1612   4.1719   3.6331   7.7545
#>    1.7549  -0.9840  -3.1662   0.8235   1.2435  10.5534   1.0228   0.3347
#>    5.6179  -5.9374  -6.7169 -13.4183   3.6784   5.6821   7.6706  -6.2777
#>   10.6609  -9.3198   3.3733   8.3848 -10.1380   0.8164  -7.4845   2.0293
#>   -5.2692   4.6907  -0.1538  -8.5968   4.0592   6.5127 -12.0751   1.6681
#>    2.3326   0.5686  11.2560  14.0028  -7.9136   0.7528  -7.7929   4.5743
#>   -6.0170 -13.4467  17.7349  -1.1736   7.2460   6.2353 -12.0200   9.4558
#>   -1.2792  13.5570  -2.4640  -8.9257   4.9165  -6.7316   7.6717   2.5841
#>    3.0801   2.2266 -13.4820   5.7208   0.2430   7.3475  15.5543  17.2854
#>    3.7720 -11.5638   3.5816   3.9917  -2.9345   8.1180   3.0167   8.3114
#>   -9.8859   5.4808   9.9585  -4.4612  -1.4848   3.1033  13.8229  -4.5329
#>   -8.4745 -15.9682   1.3945   7.1085 -11.5154   9.5478  -5.4768  -0.2199
#>   -0.6699  -7.6979  -1.2231   1.2728  -7.5408   2.5758 -12.5434   7.0323
#>    3.9763  -3.6349  14.1189  -3.6005   2.9941  -1.6028  -2.3769   9.6006
#>   -3.2661   1.3843  -1.7037   8.7203  10.3502  -1.7436   0.8987   3.1321
#>    5.8534   6.8372   3.9771  -9.7879  -8.4809   5.2292   8.3309  -3.1076
#>   -2.3502  -9.1358   7.4340   3.0671  -4.3706  15.4361  -7.9336   2.8507
#>   -5.5450  -2.1265   2.7962   2.4273   5.4109  -0.4403  -6.8531  -1.3315
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
