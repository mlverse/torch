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
#> Columns 1 to 8  -1.9832   4.3029   8.8601  -4.9779  -2.1716 -16.2312   4.0265 -18.0918
#>    8.1364 -10.4640   1.8997   3.7931   8.1963  -3.8449  -6.2190  -1.0089
#>   -2.6062  -0.2504   6.9894  12.9714   5.5424  -4.1676   1.5138  -3.5504
#>    4.8109   2.5654   9.0519  -9.3758  -5.8482  -8.4825   2.1723   7.5092
#>    4.9390   4.0700  10.7657  -8.4354  -0.5304 -16.3813   0.5583  -1.9190
#>   -5.5754  -3.9825   3.2535   2.9707   8.8032  -3.2295 -10.6453 -10.3460
#>    0.8738   0.5042   1.1751   3.1276  -5.7415 -10.7303  -8.7749  13.2365
#>    6.7865  -2.2403  -7.9097  -4.5374  -4.5151   1.9424   8.5107  18.8272
#>   -0.0670   6.4807  -3.2436  -7.3701   8.1834  16.7692  -3.4268  -7.0083
#>    0.7754  12.9450 -10.0498   3.0480   7.7334  13.3357  -0.3753   0.4605
#>   -1.8328  10.8875  -5.4325   4.7796   2.1649 -11.7340  -7.4843   7.5346
#>   -2.6533   3.6540   3.4599  -3.5911   1.0209  -9.0595   4.3952  -6.1793
#>    0.2310   4.6198  -2.9982   7.1126  -1.6592   3.1615  -1.6267  11.2766
#>   -5.4703 -12.3681  -2.3328  -7.1119   4.7877   7.1744   4.4245  -3.0914
#>    1.4087   3.9175  -4.8436  11.4865   9.2235  -4.3738  -3.5232 -17.8052
#>    4.1471   4.6465  -0.5843  -4.6787  -2.1025  -6.1617   2.7655   1.7437
#>    0.0071  -5.7126  -9.5276   5.8684   4.3700   1.4810   1.2606   7.5467
#>   -7.1619   3.2187   2.6023  -3.9962   4.8157  11.8958   1.8226  -3.3379
#>    9.3851  -3.6418   3.7079  -0.1882   6.4445  -7.8178   1.8863   6.4618
#>   -1.0855 -11.1137   9.0398  -6.3611   1.0414 -12.9293 -11.7409   3.9929
#>   -0.6041  -4.3482  -7.0186  -9.1199   3.3841   1.1312   5.8854   3.2742
#>    3.4820   4.3754  -0.1389  -8.0197   3.0409  -3.5974   8.7990  11.9569
#>   -6.7660  -3.1966   3.0361   1.3890  -2.1280  -3.9709   7.7280  -9.9127
#>   -4.4101  -1.8900  -1.8049   4.9367 -10.4395 -15.6395 -16.4365  -5.8675
#>   10.1590 -16.5738  10.3435   1.4048   6.8316  -0.4627  -1.4903  -0.2677
#>   -0.6715   6.7082  -3.5946   4.6502   0.0955  -5.8286   2.0953  -5.8525
#>   -2.7744  -3.1204 -12.2139   7.9527 -15.9383  -0.9726  -1.5132  -0.1931
#>   14.3178  -3.1308   0.4475  -5.8012   9.7875  -3.4231   0.5019   6.1677
#>   -1.6487  -0.0251  -4.8385   7.1036   0.5301  11.6203   1.4582 -10.5258
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
