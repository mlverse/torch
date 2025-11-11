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
#>  -3.6708  2.0021 -0.3258  1.1825 -1.7061
#>   2.8529 -1.3666  6.3561  0.2517 -1.7723
#>   5.4462 -0.6741 -3.9285 -5.0930  1.2643
#>   4.8807  3.4186 -1.2922 -3.7932 -4.7303
#>  -2.7511  0.6604 -0.1682 -1.1954  3.8276
#> 
#> (1,2,.,.) = 
#>   -2.3635  -2.6729  -3.6447  14.2893   1.1956
#>   -0.5963   2.2099  -2.5861   2.6098  -0.0628
#>    1.8586  10.7948  -6.6161   7.0136  -1.7508
#>    1.8889  -5.3773  -6.3196  11.9677   9.2027
#>   -1.7443  -4.5039 -11.0166  -9.5419 -11.5474
#> 
#> (1,3,.,.) = 
#>   1.0223 -4.2198  3.4122  3.6961  3.3529
#>  -3.2885 -10.2247 -1.3102 -5.6349  0.1623
#>   2.9607  3.7562  3.0365  1.9006  5.8283
#>   0.5261  3.5500 -8.2438 -2.4025  1.3188
#>   1.0564  3.3052  0.1350 -1.4746 -1.6580
#> 
#> (1,4,.,.) = 
#>   2.6959  2.2868 -1.4409  5.6321 -3.2626
#>   2.1574  6.0678  5.8928  5.1000 -2.2868
#>   2.0782  3.9290 -7.7198  1.2083 -7.2677
#>   1.2617  0.5244  1.9577 -10.5739 -1.5645
#>  -6.7475  4.7453  1.4783  6.4363 -1.1760
#> 
#> (1,5,.,.) = 
#>  -2.2510 -1.5773 -3.4723 -5.2146 -3.9564
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
