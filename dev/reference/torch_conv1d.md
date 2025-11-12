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
#>  Columns 1 to 8 -11.6212  -0.1112  -0.1560  -9.6552  -5.0135  -8.6241   8.7280  -6.9140
#>   12.8763  -0.7898  -2.3394  11.7030  -1.8053   7.0867   0.4269  -5.6224
#>   -2.4907   1.9655  -8.9514  10.9621  -3.4330  -2.1316  -7.2478  -0.8269
#>   17.9758  -0.7899   0.3556   3.3679  18.4120  -4.9976 -11.3459   3.7721
#>   -9.2813   8.2464   2.3898  -9.8954   2.9622   7.3947   1.4728  -9.0519
#>    2.7222   1.4759   8.5889  10.5686 -11.1481   5.6872  13.3211  -0.9960
#>   -3.3335   1.6273  -9.3044  13.3371   3.2308 -11.7776  -5.4031   9.0408
#>   -2.3236  -7.0376   6.7685  -0.4873  -4.4245  -6.8875  -5.1548  12.3359
#>    2.2322  10.5972   1.5133  -7.8973  -5.0728   0.6101   6.8802   1.6585
#>    3.9597  -0.2781   0.1040 -19.1494   2.3589   5.2792  -2.0225  -8.9279
#>    4.3149   5.2147   0.7695  17.1365   9.3283   8.7550  -3.9682   1.5166
#>    4.9197  -1.3707   4.1652  -6.1017 -12.5304   8.9227   0.7278  -3.2128
#>   -6.1226   7.0632  -7.2779   4.5332  -6.5448   6.4681  11.8313  -9.3886
#>   -6.6772   9.4524  -7.5463   5.2956   3.6643   9.0431   7.0637 -16.3187
#>  -12.8420  -8.0660  15.1916   9.1374   5.2240  -0.8977  -0.0191  -7.2944
#>   -8.8065  -4.6843  -0.5954   7.7450  -7.0412  -2.8312  -2.8778 -10.7073
#>   -5.7194 -10.8097   1.6264   1.9836  -3.2934   4.1196  -4.7116  -1.6189
#>   -6.3552   4.1858   3.8979  -8.4999   6.6272   9.8465   1.0762  -1.7627
#>   -0.5494  -6.6478   4.7224   1.7004  -5.7779 -14.3258   8.1147   1.2445
#>   11.5899   0.2001  -4.7057   7.4018  -0.1225  -2.5485  -2.1261   0.0551
#>  -17.0732   1.7469  -0.3866   5.7322  -9.4645  -1.6413  -9.2437   8.9287
#>  -10.4770   6.9735   5.0754   5.9922 -14.6343  -8.1061   9.8468   0.0871
#>    9.9118 -16.6402   7.5857   1.3746  -5.8153   4.9504   2.4349  -0.6035
#>    7.3170  -3.3640  -3.5362  -7.7529  11.5986  -4.2523  -4.0989   4.6977
#>    9.1914 -13.9804   3.8407   5.3343   4.1518   9.4345  -3.2662  -0.2485
#>   -9.0887  16.6374  -4.6396   0.5328  -5.7338  -3.2706  -5.7350   6.9972
#>    2.6180 -19.6459   2.0160  -4.9693   2.7364   4.2272   6.2250  -4.1845
#>  -10.5277  -4.4460  11.8691   1.4482  -9.9284   6.4988   8.5709 -10.7616
#>   18.5592 -14.1128  -3.2995   1.0710  -1.9412   7.0391  -0.4469   2.7908
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
