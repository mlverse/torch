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
#> Columns 1 to 8   9.5931  -7.8835   8.9728  -2.1578  -4.9302  -0.9256   1.2946  -3.5130
#>    2.2998   3.0737   3.1876  -6.5607   6.7704   1.2456   9.9046  -2.0202
#>    4.0477 -13.5921   9.4007   1.8533   1.9714  12.6200   0.3894  -4.9326
#>    6.5334  -3.7008   2.0910   0.0098  -4.9545  -1.9813  -3.4059  -5.4052
#>    2.1504  14.4911   2.1222  -2.7170   2.7073  -1.5095   9.2460  -1.5324
#>   -9.7726   6.3373 -11.3960   7.4387   1.6209  -9.1345  -7.9794   5.5311
#>    1.8275   6.7269   7.9181  -3.1820   1.5178   3.1408  -3.6534   6.5707
#>    0.4932   9.9720  -2.9348   2.4767  -4.1745   2.3630  -3.3933   5.1903
#>    3.3474  15.9606   3.0185   6.3608   0.5127   4.2083   0.0464  -6.2017
#>   -2.8116 -10.3296   0.6668  -3.2949  -3.9810   2.2544   3.6527  -4.7989
#>  -14.6966   8.8695   1.3237  -6.8173  -1.3779  12.7851   7.6538  -0.4625
#>   -1.0765  12.6675  -6.4197   8.6272   2.5824  -5.4841   3.1040  -4.8028
#>   -0.3292   4.0704   3.8432  -5.1623   2.7248   6.5608   0.7988  -0.6557
#>    8.9905  -4.6922  -1.8520  -4.1188  -6.8890  -1.4568  -3.7811  -2.8779
#>   11.0365  -8.4843  -5.2954  14.6005  -2.6770  -6.0100   2.0496  -1.4512
#>    1.2443   0.2034  -0.2451  -1.2705   6.9672   4.6270  -2.9580  -6.1738
#>   -4.6169  -5.3360  -5.8059   1.6999  -9.9409   5.9988  -6.1170  -3.2507
#>   -3.6105 -12.4903  -0.8041  -8.7811  -3.4846  -8.7113   0.9494   1.3360
#>   -3.5656   0.4610  -8.7431   1.0824  -0.6439  -9.2273  -7.0345  -3.7739
#>   -4.8795 -17.5458   1.0220   1.8733   6.4301 -13.2276   4.1554   6.2307
#>   -2.6494  -5.3041   0.1399  -0.9405  -1.7168   3.1547  -7.0168  -1.1389
#>   -7.5794 -11.6843   3.0080   8.1006   1.8369  -3.9589   1.9046   4.3819
#>  -15.8415  -8.6908  -3.6859 -11.1640   4.6654  11.1461   2.9586   3.5769
#>   -0.7414   8.5652  12.3612  -5.3661   6.2098   6.6230  -0.6242 -13.1381
#>    4.3687  -3.8544   0.0278  -0.5015  -6.2462  -8.2327  -4.6240  -1.6846
#>    6.0389  -3.0561  -1.8480   2.3246   7.2590  -0.0046  -3.4641  -7.9990
#>   -8.6471  -4.6001   1.6160   6.5029  -9.2530   2.8495   1.7108  -5.5432
#>  -13.4515   3.4530 -10.1733  -7.7749   2.4026  -0.1103  -0.1653  -7.0649
#>    3.9803  -3.2824   1.6261   9.9066   1.1579  -7.1485   1.7050   2.2058
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
