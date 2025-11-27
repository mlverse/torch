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
#>  Columns 1 to 8  -8.4966  -7.1858   0.9130  -3.0228  10.9175   6.7176  -5.6526   0.2722
#>    5.0855   4.2095   1.0102  -9.1928 -12.3300  14.6966 -13.1008   3.1633
#>   -7.3335  -6.8660  -9.3375   1.3890   0.0113   8.8025  -4.3081  -6.1265
#>    6.9159   9.4168   5.7925  -7.1778  11.4640   7.9874   6.0945   2.7619
#>    5.9137  -7.0552  -3.8385  -3.5954  -6.2247   0.1455  -2.1283  11.3307
#>   -6.3420   6.1467   5.6650  -2.3693 -11.9185  -2.8997   0.3573  -8.9534
#>    9.9910  -4.2357   8.1486  -5.1256 -13.4995  -7.3680  -6.9391  -1.2263
#>    4.5903   4.7300  -4.3853  -9.4487  -0.8985   1.9231   0.2031  -9.8015
#>    4.1591  -2.3942   1.7032   7.3598  -0.9426  22.2192  -4.0292   6.5468
#>    9.3673  -4.9640 -13.8824   5.2249  -1.7049   5.3551  -2.3983   9.9828
#>   -5.5147  -8.5551  -6.7621   0.2688  -3.7489  -0.6868  -5.4656  12.0405
#>    3.2600   8.0775   5.3034  -2.8837   0.4610   3.8522  -2.9062  -5.8448
#>    5.8935  -1.3344   1.8896  11.8041  -4.5955   3.9419   5.0773  12.5437
#>    4.6997   0.6122   2.8464  -6.7180  -6.9090  -1.2711   5.4290  -2.0010
#>    0.3502  -6.7935  -9.3208 -11.6004  -3.3836   7.2076  -3.3198  10.0514
#>    1.4956  -9.1865  -7.3921  -6.6026  -4.4098  -9.3439   8.5201   3.6289
#>   -8.8272  -4.0299   2.7032  11.8433  -1.1951  -0.7462  -4.1740  13.4981
#>    3.2504  11.3708  -1.6011  -2.7463   7.3870   5.7883   3.3163  -6.9714
#>   -9.9118   1.3825  -6.0762   2.3500  15.1346   2.0370  11.2094  -3.2597
#>   -0.8396   8.3478   2.3615  20.8776   3.1111 -14.0111   6.9886  -3.0690
#>   -8.4584  -0.7478   4.5949  -3.9159   1.2939  -3.6316   5.9912  -2.7671
#>  -10.1268  -7.4071  -3.2424   7.4021   4.0543   8.6127  20.2994   0.5807
#>   -0.8135  15.2657  -1.6950  -8.0112  -2.5020   7.9158  -9.5962   2.6011
#>   -2.7145   3.7322  -3.9274 -10.6393   3.0118  12.1689   0.8103  -2.3582
#>   -1.1800   0.8562  -3.7549  -6.1303   5.5453  -3.6943  13.7839  -8.9031
#>    0.0287   0.9243   6.2790   1.9165  -8.5072   2.9399  -3.0392 -12.2447
#>   -7.1939  -1.8254  -8.4213   5.2014   4.5608  -0.6144 -11.8875   8.1975
#>  -15.6981  -3.5114 -15.6108   3.9859  18.6188   2.0643  -1.5925  18.3159
#>    3.7971   5.3294   6.5835  -2.3748  -5.4427  11.0453 -10.0979   2.4698
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
