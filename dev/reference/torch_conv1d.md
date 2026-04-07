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
#> Columns 1 to 8   3.3621   3.1927  -0.0844  12.7255  -1.1041   5.7810   5.4771   3.8285
#>   16.8716   1.7424   0.5687   5.3526   0.9251   1.3464   6.0860 -10.1362
#>   -2.7918  -1.8370 -12.9988  -8.7058   8.7812  17.0191   7.7346   7.4556
#>    5.2514   1.5234  -0.7532   3.6843   7.7774 -16.5307  11.1846  -0.6126
#>   -1.1729   1.1585   2.0003  -5.7797   1.0369  -1.0424  -7.6168   0.4635
#>   -0.3758   3.1014  -5.1121   0.6644  -2.1803   9.4790 -11.4191  15.5637
#>    2.6402  -3.8942   7.2064  -9.6370  -0.7693  11.4354 -11.7298  -0.2989
#>    4.7174  -6.7725  11.6135   1.2554   1.1001 -14.0418   6.5232  -7.8759
#>    6.4635  -1.3320   2.5170   6.1733  -2.6273   4.7671  -3.0391  -5.0680
#>   -0.3885  -0.9739  -5.7783  -5.5182  -2.7447 -17.4261   0.0941   2.2386
#>    4.7166   1.2687  -9.7492   5.2450   2.1403  -6.4982   3.3007   0.5275
#>   -4.3122 -16.0652  -2.0453   6.1510 -11.9672   8.2024   4.5367  18.1667
#>   -1.8422  -3.0407  -0.2855   8.9263   3.0503  -6.6609   5.2650  -8.5751
#>   -2.3355  -4.3191  -9.2577  -3.4821  -6.8781   6.0764  -5.2904   5.7564
#>    5.6566   8.2285  -1.1528   3.1432   8.7242   0.5181  13.4067  -3.8727
#>   -5.6566  -0.0099 -11.5906   2.9283   2.5001  -8.3253   0.0107   5.0633
#>   -2.0518  -0.0117   1.0714   2.2990   3.9718  -6.1293  -7.8129  -0.5774
#>    6.5258  -2.6577  -8.4882  11.2476 -18.1432  -1.6142 -11.6878   8.5972
#>    9.7136   0.7357  17.3917  -5.9983   7.6901   2.3082  -9.6240  -6.9891
#>  -19.3550   5.5719  -7.7897 -12.7573  18.2049  -2.7533 -10.9958 -13.9643
#>    4.6856   5.9432  -8.2324   3.0243   0.1690  -2.7997  -3.3313  -0.7576
#>   -0.1294   7.3543  -0.7984   1.6089  -1.2356   4.5218   5.9142  -5.6175
#>    4.4965   8.5839   5.5928   2.6743   2.1708   1.6949  12.1222  -0.4590
#>    7.8385  -2.0181  -4.2190  -3.4547  -2.1514   9.8673   2.5868   0.2441
#>  -15.5448  -7.2803   1.0183  -5.0376   2.6040   1.1254  -5.5022   2.0238
#>    2.6355   9.3863   3.8466  -1.6964  -8.2050  -9.9814  -1.1147   5.3764
#>   -0.2298   2.3277  10.7823  -8.7813   5.0089  10.7155  -7.6440  -2.3550
#>   -1.0272 -11.9721   8.3092   1.0896  -8.9380   6.1234  -4.4969   8.1008
#>   -6.3469   5.4836   4.9217  -0.8349 -11.9351  -4.7313  -5.3626   0.0703
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
