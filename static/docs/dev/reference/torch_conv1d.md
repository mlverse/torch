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
#>  Columns 1 to 8  -0.0399   2.0103   0.4517  15.1908  -1.1191  -1.5297 -12.1638   8.6864
#>   -5.5948   5.5661  16.0985   0.0155  -1.5606  -5.4577   9.8086   4.3042
#>   -8.9633   9.4667   5.5732   4.2461 -13.3492  -0.7253  -5.3176   4.0344
#>    9.3708   6.6012  -0.4189  -3.8606  -3.9645  -3.7989   1.1312  -4.4634
#>   -4.8505   2.5361   1.4357  -8.6775 -15.9163   1.4252   6.8673  -5.0086
#>  -14.2326  -2.3305   3.2338   2.3309 -16.3477  -0.3910  -1.3698   4.8824
#>  -10.2347  -2.5137   6.8602  14.7599   8.6304   4.8794   4.1813  -2.7304
#>   -0.5004   3.1624   3.7989   9.7287  -3.7016   1.2229   5.8732  12.7473
#>  -17.1445  -7.2170  -0.3768  -2.5236   0.3070   4.4261  -2.5914   1.5759
#>   -2.3926 -15.6308   6.2715   2.9263   1.7976  -8.3703  -4.5590   1.6036
#>   -7.2016   6.4179   4.0121   8.2774 -10.2834  10.8593   6.5570  -3.2228
#>    9.2597  -1.7946  -5.3953  -2.5220   6.4843   5.5871  -6.8019  -8.2042
#>  -12.7481   3.9039   1.6388  -1.7845  -7.3581  -2.5724   9.9685   6.8365
#>   -2.9142   3.3853  -9.8467   1.4089   8.4595   7.8975   7.1076  -2.2674
#>   -7.0895   7.0029   1.1979   8.2067  -3.8954   5.2143   2.3170   2.9599
#>   -8.4130   6.7045   5.6058  -7.7266  -2.0260   1.9072   5.0485   9.8567
#>    0.0343   2.7437   0.6689  -4.8464  -1.5168  -2.3826  11.6427   6.2501
#>   -9.1438   4.4151  -8.5392  -3.2576 -12.7474   2.1530  -5.0732  -3.8746
#>    6.8293   1.2915 -10.3509   8.0883   6.2695  15.8727  -5.1236  -8.3285
#>    4.8102 -14.6736   4.9817   4.9200  -1.6072   0.2609  -0.6845  13.4829
#>   -5.2866   0.6429  -7.3959   2.8726   2.5119  -8.0737   4.3943  -2.7086
#>   -5.9458   7.2573   9.9741   2.1623  -2.7931  -8.9127  12.7288   7.3563
#>   10.0288  -5.4266   2.2129  -9.2093   6.4459  13.3688  -4.7323  -1.1782
#>   -3.7032   5.0677   5.7248   0.9998  -9.0854   5.4754   2.9469   7.9141
#>   -0.2021  -5.1664  -8.9887  -4.2022   8.3852   1.1330  -8.8332  16.8060
#>   -3.7556 -18.7530   1.1319  -1.7102 -10.4763   3.0764  -6.4131   3.4932
#>  -12.6173   5.5721   8.5958  -1.5879  11.2390  -1.3916   3.5088  -8.5058
#>   -0.5530  -0.2437  -1.6881  -2.8429  -0.5142 -16.2823   1.8826  -8.5802
#>    0.6049  10.5722   1.3604 -11.4045  -2.2833  -4.4657   5.1742  -6.5052
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
