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
#> Columns 1 to 8   0.5141  -0.6793   1.1099   3.0029  -4.3537  -1.2871  -3.0970 -10.6292
#>   -3.9989   5.3361   4.8532  12.1057  -2.7493 -12.0813  -5.6340   8.5318
#>  -13.9888  -4.1509  -4.2034  -2.4267  -1.8208  -3.8631  -8.3934  -6.8825
#>   -0.8801  -2.7000   0.8181 -12.5708  -8.7692  -2.4254   1.4059  -2.3366
#>    6.7515   2.7246  -4.6816   3.1732  -5.9617  -2.2588   1.5027  -8.6424
#>    0.0207   6.2832   6.8336  -9.3054  -4.9821  -1.3693   8.3810  10.1333
#>   -2.2242   5.0603  10.0662   1.2646  -7.1119   2.9110  -1.6288   3.3204
#>    6.3169   7.9539   0.5069  -4.6015  -5.4283   4.7464  -0.1210   1.9266
#>  -13.3824 -11.6759   5.6829   0.0744   1.0601 -13.2933  -2.2121   2.3244
#>   -3.9938  -4.7551  -1.3071   3.0856   4.7458   0.7560  -5.1428  -5.6986
#>  -10.0575   3.6945  -5.1563  -7.3369   6.6174   6.7889  -3.0244  16.7625
#>   -2.5605   4.4775   3.1659   5.1224   4.4729  -3.2733  14.4482  -6.7773
#>   -8.0982  -0.3172  -3.6394   2.9918  10.8038   9.3698  -7.3399  -1.5632
#>    6.1428  11.8041   6.0185   5.8669  -9.6068  -0.4689  -9.1451   2.7200
#>   -1.4093   2.2385   3.0887   0.4437  -3.3268   0.2122  -3.0802   3.2533
#>    6.8890   0.1862   3.1789 -13.2235 -10.7065   4.6368   0.6225   1.3035
#>    5.2010   5.7934   2.1780  -1.1450   3.4048   4.4663   8.5329  15.7947
#>   -1.4854   6.6009   7.3778  13.2150   6.7781  -2.4892  -4.4063  12.0660
#>    1.5335 -12.5912   0.9745   1.8040   1.3446   4.0309  -7.5952  -0.2106
#>   -0.1377   4.6484   8.0466   7.2013   4.5734   9.7553   7.1202   3.6472
#>    2.9209  -4.0756  -0.2212  -7.7486  -4.2947   1.1094  -0.5762  -4.5731
#>    1.4090 -13.1916  -3.5341   2.8243   1.2767   2.0397   6.5902   6.5966
#>    5.2341   0.8310   6.0164   5.5909  -5.0245  -0.9678   1.7573  -6.0049
#>   -1.2416  -7.2246   2.6069   1.1887  -2.6971   2.5533   8.9113  -1.3866
#>   -9.4506  -0.3615  12.0186   3.7943   0.3102  -6.2996  -7.9290   2.8672
#>   12.5949  -0.8889   5.5326   8.2273  -2.5189  -1.2462  -0.9786  10.2234
#>    3.3965  -7.7975  -0.3783   5.5607  -8.5550  -2.6496  -2.1886  -3.5714
#>   -9.6979  -0.9559   9.6314  -7.4126   3.2022  13.6930   8.9120  -8.0221
#>    2.8768   6.8652   6.1597  10.1047  -7.2783  -3.4522 -13.2838  -3.3294
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
