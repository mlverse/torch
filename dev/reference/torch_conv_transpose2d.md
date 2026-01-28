# Conv_transpose2d

Conv_transpose2d

## Usage

``` r
torch_conv_transpose2d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padH, padW)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padH, out_padW)`.
  Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

## conv_transpose2d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
inputs = torch_randn(c(1, 4, 5, 5))
weights = torch_randn(c(4, 8, 3, 3))
nnf_conv_transpose2d(inputs, weights, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#>   7.9947  2.1870 -1.6665  2.4940 -1.3914
#>  -5.4752 -3.1150  2.3552 -0.4124 -3.3087
#>  -3.2994  0.7120  0.9118  3.8505 -2.0632
#>   4.8103 -10.1796 -5.6329  5.7254 -0.7365
#>   1.6402 -1.3006 -4.7610 -2.7414  1.8612
#> 
#> (1,2,.,.) = 
#>  -4.8017  7.6925  0.5434  2.4439 -0.6430
#>   6.5432  3.0845 -0.0238 -5.8415  1.1678
#>   0.2119 -7.1479  0.5061  0.9148  0.0570
#>   4.3818  6.6153  5.4112  1.9871  4.1599
#>   5.1445  2.1850  3.3923 -1.7928 -0.8157
#> 
#> (1,3,.,.) = 
#>   -1.2873   6.5095  -6.5458   4.7033   2.2477
#>    5.1944   0.9526  10.3252  -6.3227   0.2455
#>   -2.9409   4.6255  -2.9013  -1.1478   1.0946
#>    1.1065   0.4402  -8.4444  -2.2300  -2.1868
#>    0.8045  -2.3128   1.6822   1.2559  -4.6950
#> 
#> (1,4,.,.) = 
#>   -1.2971   4.1752  -2.9092  -2.2611  -6.2658
#>    4.6882  -8.7063   2.7609   5.2225  -3.5175
#>   -3.9785  -0.6347   4.3025  -4.9424  -0.9844
#>    4.4247   1.2835   5.7031   2.1815  -1.2635
#>   15.8753  10.7158   2.5478   2.4206   1.7452
#> 
#> (1,5,.,.) = 
#>   -4.4556   2.5469  -2.9017  -0.9058   0.3388
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
