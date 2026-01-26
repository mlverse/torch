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
#>  Columns 1 to 8   6.1113  -1.3416   2.5537  -0.3511  -3.7622   9.4438  -9.1623   5.4123
#>    0.1384   5.0861  -0.4446  -7.1778  -2.0304  -4.7200  -2.3110   7.0517
#>    7.3361  -6.8946  10.6440   6.8492 -15.2634  18.2610  -1.1591   2.4721
#>   16.9490  -5.1302  -2.9508  16.2452  -5.0292  13.2416 -13.0098  -6.0641
#>    5.5661  -7.3467  -6.1406   4.5786  -0.4492   0.5630   1.9214   0.3948
#>   -2.9552   1.0690  -8.1115 -12.0575   2.1286   2.3253  11.3612   7.3762
#>    9.6896   0.8754 -11.9025   8.4991  -2.6257   2.8595  -1.9257   1.6814
#>    1.9446  -1.3896   5.8513  -7.8567   2.9452  -1.9360   0.6543  -4.4542
#>    1.2410   1.8889  -8.2458   0.0838  12.1690   0.1441  -7.4621  -2.1215
#>    4.5208   8.2307  -6.3371  -1.5987   0.8926  -8.3113  -2.3910  -7.9629
#>    3.0710   5.5355 -16.9383   7.2080  -5.6063   9.7497   1.0945  -2.6167
#>   -5.4261   6.8228   3.4538   6.6948   1.2137  -0.5459   9.7850  12.7843
#>   -2.0128   3.9406  -3.3764   1.6644   1.7191   0.1825   5.0427   0.8231
#>    7.5926   6.7747  14.7720   9.6155  -4.2885   1.8971  -3.4543   5.2117
#>   -0.1593  19.0087  -9.6421   8.6122   1.2728  13.1211   2.7347   4.5025
#>   -0.7439   1.4071  10.5095  -4.1528   4.0271   0.8212  -6.4963  -8.9867
#>    0.4737   5.8617  -4.6700   3.9658   4.2204  10.7399  -6.1406  -3.1327
#>   -0.5139  -2.3034   2.0571  -4.5131   6.7677  -0.5331   4.1831   7.5730
#>   -5.5786   6.6846   5.2202   2.0031   3.5624 -16.2284  -0.1967  -4.3535
#>    5.6427  -3.7625   7.6101  -0.5925  11.9024  -5.7443   8.6710   0.1619
#>    0.0352  -1.5355   5.5508  -0.1707  -8.0384   3.7810   0.8357  -7.7918
#>    3.5590  -0.2908   3.8150   0.1858   7.3002  -2.0670  -6.1812  -3.9409
#>   -0.5022  -7.2382  12.0235 -10.2434   1.1811   0.0447   2.4838   1.1117
#>    1.9257  -0.7390  -2.7015  -1.3446  -3.5291   1.2072  -1.4144  -2.2389
#>   -0.7546   5.1803 -10.4647  -0.4777   4.6357   3.0945  -1.1900  -5.8668
#>   -8.0224   7.5505   4.0480  -1.8914   1.4173  -7.3396   0.6771  11.8458
#>    2.6614  -0.5376   0.2340   6.0813  12.8008  -6.4823   4.1757   3.5352
#>    8.8455  -8.3208  -7.2307  12.7220  10.8836  -0.0164   3.9549   0.9624
#>   -1.1724  15.1027   3.7021 -13.4794  18.9815  -6.0994   3.6842  -5.6703
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
