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
#> Columns 1 to 8 -10.3514  -6.1694  -7.9181  -1.8363  -8.5330  -8.1242   5.2754  15.0595
#>    0.2167  -6.1105   9.1794  -6.9325   1.1064  -2.0435  -6.8925   9.1412
#>   -3.3079   5.1453  -3.5225  -2.6615  -1.7681  -8.5782   7.3865  -7.1532
#>  -11.2184  -5.4670   5.2383   2.3012  -6.2874   3.2165   6.0730  -1.7347
#>   13.6846  -8.3035   5.4806   0.2768   4.2486  -1.3807  -8.0871   1.1472
#>    2.9110  -3.6274   7.8208 -11.5255  -8.7015   7.9041   6.7703  10.8618
#>   -7.1826   3.4899  -5.9035   6.9798  -8.7133 -15.3162 -12.3097   0.7969
#>   -4.1232  -5.4782  -1.6888  -7.6133  -8.4446  -2.9564 -11.0829   5.9577
#>    7.6163  11.1039  10.3484   9.6289  10.2139 -10.8235  -2.4854  13.0825
#>   -6.3203  -0.2579   5.5815   3.6706  -6.8145   1.1416  -3.7066  -6.5986
#>   14.1796  -6.7610  -5.4302   1.7932   6.9443   0.9413  -3.4150  -7.0634
#>   -7.2689  -0.9857 -10.8289  -9.9017  -0.7706 -11.6643  -2.4308   5.6915
#>    6.4935   8.4905   3.0556  -3.7695  -6.6213   0.3711   6.9220 -14.0509
#>  -11.8663   4.3545   7.7113  -2.8937  -8.2584  -2.0951   5.7423  -2.1090
#>   -2.4459   5.6665  -1.2524   4.5713  -0.9564   5.9134 -13.8533  -1.0117
#>   -3.9973  -4.6422 -12.4053   4.0687  -6.3372   1.6606   1.6982   0.6939
#>    8.7754  -0.2330   0.2369   5.3978  -0.0518   1.4221   2.0950   0.3126
#>   -4.4686  -0.0795  -1.4513  -3.9189   0.9556  -6.5556   4.1566   1.9853
#>   10.4952   2.1626   2.0039  10.4369   4.2549   5.9570 -12.5194  -9.5479
#>    9.1590  -8.3136  -4.5920   7.1437   6.6857   0.7786  -3.2928  -8.4669
#>    4.4433   6.0276  -2.9468  11.7136  -6.7512   1.8446  -2.6269   1.5335
#>   -3.4595   5.6794   7.1583  -2.1723   0.4713  -1.6195   5.9674   6.3319
#>   20.4159  -7.2290   5.7624  -6.4558   9.3896   0.5796  -5.6873  -1.6043
#>    2.5524   9.2403   5.9080   7.9042   2.6518  -0.5658   1.3508 -12.4148
#>   -0.1501  10.1375   1.6371   6.9743   6.7250  -1.9283  -4.9632   1.7482
#>    5.1677  19.0617   0.5731  20.3938   3.5915   3.0229  -1.3575  -6.0852
#>   -9.4599   2.3029  -1.3639  -0.0926   2.5959   9.1996  12.2947 -11.6291
#>    2.8422   1.1771  -5.4053   6.3555  -1.1341 -11.0190  -3.6121   5.4747
#>  -16.8454   0.7726  -2.0606  11.2209 -19.1595   4.9702   7.1573   6.2492
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
