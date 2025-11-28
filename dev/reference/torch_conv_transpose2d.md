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
#>   0.2202 -3.4398 -4.5012  1.6079 -2.2903
#>  -4.1291  4.3575 -6.1121  6.0099  4.2854
#>   2.4690  0.8682  5.5854 -0.4782  0.8962
#>   3.5572  1.3954 -1.1118  0.6686 -0.8776
#>  -2.0699  0.8051 -3.8975 -3.1484 -2.2480
#> 
#> (1,2,.,.) = 
#>  -0.8023  2.5599 -11.1332 -4.9716 -6.4580
#>  -3.5828  5.8840  2.3181  6.1382  3.7109
#>  -2.4545 -5.0782  1.8882  5.0257 -2.8389
#>  -9.5117  4.2036 -3.6803  5.3974 -3.4833
#>   4.3141  2.5592 -0.8665 -0.9070 -2.5316
#> 
#> (1,3,.,.) = 
#>   9.2229  1.9271  4.1989 -0.0923 -0.5306
#>  -4.8440 -6.4411 -5.2166 -3.4311 -0.3318
#>   5.3565 -2.9130 -0.6576 -0.1732 -3.1689
#>  -5.5128 -4.4926  4.3473  1.4318 -0.9423
#>  -2.2125 -3.5358  2.2468 -0.2992  1.8471
#> 
#> (1,4,.,.) = 
#>  -1.8277 -9.4388  6.3965 -5.0485  2.8237
#>   9.2644  0.9155  1.5128 -4.7280 -4.6392
#>  -6.6516  2.6736 -1.0770  2.4618 -3.3288
#>  -2.4828 -2.2880  2.6717 -7.2096 -0.1579
#>   0.3267  1.8994 -5.2506  0.4390 -4.4199
#> 
#> (1,5,.,.) = 
#>    6.6482  -4.2423  -1.9953  -9.0472   6.7436
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
