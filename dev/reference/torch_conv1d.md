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
#> Columns 1 to 8 -15.4785   3.8274  -2.9875   5.6942  15.9873  -1.8506  -4.0553  -7.5230
#>   -3.7380   1.2783  -9.4074   1.9046   2.4194  -2.5578  11.5908  -2.1335
#>   -6.0639   8.1948  -7.4296   6.1595  -2.1487  16.7692  -9.0000   0.8040
#>   -2.4447   0.8885  -1.1754  -1.8863   1.9921  -1.9481   1.6936   3.1055
#>   -0.0101  -2.0281   5.7849   0.6224  -6.0918  -5.7307  11.4859  15.3498
#>   -9.2952  -5.8737  -4.3412   2.0344   3.7113   0.1651  20.1128  15.6136
#>   -0.6453  -9.6636  -4.2543   2.1132   1.9127  -0.1547   6.5017  -1.3726
#>   -3.9099  -6.0223   8.7368   1.8849 -14.3517   7.2184   6.5481  11.4886
#>    5.6233   2.1795   8.0709   5.2198   6.4877  -1.7449  -2.4830  -8.8644
#>    6.5271  -5.4966   5.2405  -1.2812  -7.1112   1.0266  -2.4575 -13.1710
#>   -0.3276  -0.3626  -2.2032  -3.3232   2.8249   2.9340   1.6682   8.4216
#>   -2.2350   4.0791   1.0136  -2.8168  -2.8378   9.1138   6.0994   1.4849
#>   11.4061  -9.4854   1.0871 -11.1557  -6.5292   3.0731   9.3996   4.2456
#>   -2.3111   0.4802   2.5344  -3.7158  -2.1378  -5.8507  12.6121  10.9581
#>   -5.8051  -5.8222  -3.7180   2.7173  -2.2893 -15.5099   1.4685  -2.3687
#>   10.1625   8.0752   5.0044  -4.0531   5.1133   9.2230  11.7474  -1.3390
#>   -4.0017  -1.6222   2.8218  -6.6522  -4.0909  -0.9985   5.0177  20.0046
#>    6.5832   7.4902   1.1658  -2.4942   3.8916   5.4937  -3.5047   2.0512
#>    7.9539  -1.6576   2.3061  -0.5350  -8.0878   1.9721 -16.1258  -7.0351
#>   10.0619   1.2105   1.1555  -8.5949   0.0955   7.1962  -3.5664  -2.9925
#>   -6.5627   4.4141 -16.6935   6.3846   4.7057   1.0324  11.2399 -10.2720
#>  -14.9097  -3.4269  -8.1339  -1.2665   2.8838   9.4472   7.8771   1.1333
#>    0.9314  -2.7844  -2.5896   1.7870  14.6852  -3.1382   4.8112  -8.2407
#>   -6.8819  -5.9037  -7.9022  -1.5308  11.1538  11.6865  10.8499   3.5203
#>   -8.2750   8.1822 -11.8454   3.9088  -9.4604  -7.7446   4.9332   0.4030
#>   -9.9945   7.9054  -8.0250   9.6087   0.2130  -9.1061  13.2501  -0.7107
#>   -7.8663  -3.1592  -4.7837   0.6253   2.3536  -3.8905   3.7677   0.4038
#>   -3.5979  -0.5784   7.9045   5.6077  -6.2476   0.9619  -5.3527  -1.0866
#>   -1.9045   1.9600  -3.5679   2.4648   0.2362  10.5532  -3.8628  -0.3689
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
