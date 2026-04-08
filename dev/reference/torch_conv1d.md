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
#> Columns 1 to 8  10.3787   0.3451  -0.7581  -2.9398  13.2078   0.7137   0.9541  -4.9122
#>    9.1338   6.4161 -15.1092   5.3843   3.8103   0.1671   1.2929  -5.8041
#>    0.2129  -5.8046  -2.3327  18.3167  -3.8333  -1.3679 -28.1050  -0.2130
#>    2.1660  -6.3014   6.6867  -0.7032 -12.8307  -5.8667 -11.8387  -5.2384
#>   -0.6651   4.3373  -0.3924   4.8213  -4.2623  -5.7693   9.0028   2.1294
#>  -12.3736  14.3075  23.7919   9.8930  -6.6032   5.5221   5.7779  -6.4676
#>   -1.9331  -3.3049 -10.6846  -2.8708   1.9319   0.1267  -1.4010  -5.5562
#>    1.5895  -2.1689  -1.8895  -5.9543   7.2993 -12.9930  -7.6286 -14.3373
#>   -2.7251  -7.7388   6.8628   6.2150 -17.6958   5.7120   3.9229  -0.6611
#>   -7.2157  -2.5324  14.4963  12.2838  -0.8935  -3.4033  -3.9704   5.6092
#>   -5.4180  -3.6679  -0.7782  -6.5565   4.1691  11.3501 -12.7112  -0.8418
#>   -3.5609   2.9971  -5.4873  -4.8163  -9.3790   5.8835   4.6681 -12.2309
#>    1.1924  -0.4622   9.3682   6.3832   6.4670   4.1639  -3.8066   5.7658
#>   -9.9135   2.0612   4.2232  -4.2721  -3.9279  -0.2592   2.6238  -5.0159
#>    2.6482   0.8230 -13.8643  -6.6353   1.3208  -8.1502   8.3734  -5.5806
#>  -12.6208  -3.3187   0.5041   6.6439   7.6717   7.6651   3.0030  16.4714
#>   -2.6730   7.8699  -6.1726 -10.6161  14.4808   2.8292   8.2935   1.9556
#>   -7.4662   4.6913  -5.5767  -6.8521   0.8749  -7.9829  -0.1651  -2.9280
#>    8.0910   9.1733   2.5511  -4.5439   8.6234   1.9609  13.8980  -1.1429
#>   -8.5347  -1.6080   4.4969   2.0115  13.1973  12.3669  -4.4268  -1.8737
#>  -16.6016 -10.8322  -4.2974  -2.0027   5.3539  -3.3206  -4.3386 -13.6024
#>   -7.1125   9.8391  -7.2760   4.5921   1.8143 -11.8508   3.2331  11.4101
#>    7.9073  -2.0569  -1.5185  -3.3339 -10.1281  -1.3453   6.3123  -7.0652
#>    1.8302   5.6924  10.0035   0.0765  -8.7790   1.8165  -5.3992  -3.0050
#>   -2.4825  -0.8214  19.4268   1.0910  -6.8107  -8.4031 -10.9136   7.3696
#>    9.2178  -6.6801  -4.6354  -3.8069   2.8316   3.5115  -1.1569 -11.1959
#>   14.4111   4.6091   9.0254   3.0381   5.4227  -4.2993  -0.3986  -3.1061
#>   -2.8979   9.5830   2.0674  -6.9761  -5.6239  -7.2046  -8.6758  -0.1876
#>    3.4198  10.3226   0.1482  -2.0448  18.2200  -5.4311 -11.9890   3.7364
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
