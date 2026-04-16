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
#> Columns 1 to 8  11.3935   8.7605   3.4353  -3.8662   1.0909   0.7271   1.1482   5.7271
#>   -3.7576  -9.6805   4.5002   3.7608   2.9370  -8.0134  -0.9286  -3.8094
#>   -9.2029   1.3359   0.2721  -4.1372  -0.4264  -5.0695   5.8283  -5.7783
#>    2.8689   4.6905  -1.3935   2.8472  10.0675   4.9799 -12.6480   7.1463
#>   13.0285 -13.3040  -8.1384   0.1550   1.8897   7.0382   2.4882   8.8575
#>   -4.5488  -7.2495   1.7973   5.4374   3.3084   6.0883  -2.7539   3.7758
#>  -13.9742 -11.2115  -2.0749   0.7014  13.4930  10.6288   7.7954 -11.0260
#>  -14.3090   4.5032   3.2444  -0.2493  -6.4901   4.4486  -8.9912  19.9328
#>    3.5430  13.7494  -8.1171  -1.1369  -0.9669   6.0821   4.9141  -1.9091
#>    1.6987 -13.2541  -5.1052  -3.6259  -6.5869   1.4980   2.8001  -5.8391
#>   -3.3064   8.8768  -0.6006   2.2132  -7.6755  -8.6815   8.8563  -6.1131
#>   13.2221   2.1821 -10.5541  -4.1564   4.4709   0.5544   4.5812   4.1565
#>    6.0512   6.6593  -2.9607  12.5524   2.2129   0.9922  -6.6735   2.8846
#>    9.2980   1.3254   2.8599  -2.2226  -3.4944  -0.2364   6.7287 -10.6537
#>  -12.4522   2.5031   3.6753   1.7499   4.1923  -1.8615 -12.2886   2.1756
#>    5.5566  -7.6746  -2.0820  -0.4580   1.2585   0.6464 -13.3133  19.5569
#>   12.3722  12.7527   3.5776  -2.7227  -0.4699 -10.7796   4.6882  -2.8825
#>   -0.6998   7.4784  -7.7227  -3.4383   6.4209   2.6632 -10.4554   8.4890
#>   -5.3572  -8.1251  -0.1243  -8.4257   6.6157  -1.6409  12.0447 -10.5422
#>   -4.4263   1.7732   0.4621  -2.0090   0.0057  -9.0813   3.2065  18.6600
#>   -6.6612   0.1331  -0.2507  -5.4827   6.1560  -0.0792   2.3499   4.0190
#>  -13.0445  -0.8864   1.0167   2.0044   5.8495   3.6557  -2.0725  -5.8523
#>   -3.0801   1.6908   5.0328  -4.0192   4.8205  10.2474  -6.2701  16.6543
#>    3.8001 -11.5628  19.4197   4.6876   0.1885  -4.7561  -1.3221  -9.8526
#>    1.2894   0.6539   9.0664  -3.2124  -5.3594  -0.3601  -6.1897 -15.0331
#>   -9.7705   8.3338  20.8277   4.5401 -14.9192  -3.5753  -2.9988   1.0974
#>    1.4788  -7.8278   1.4317  -8.6244   4.6906  -7.2197   1.1608  -5.4379
#>    1.8757   4.9990   4.0168  -1.3839  -6.6416   4.1431  -9.1723   1.2686
#>   -8.7151  13.8415  24.6354  10.9577  -0.6637  -5.1414  -5.3691  -6.9883
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
