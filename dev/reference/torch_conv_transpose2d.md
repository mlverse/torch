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
#>   -0.5075   2.0867   3.5409  -7.2235   3.5032
#>    3.9265  -0.7814  -9.6196  13.0156  -2.8309
#>   -8.1003  11.0458  -1.1141  -4.6186  -1.4607
#>    5.8437   1.3351   5.9440  -0.0737   4.0442
#>   -1.1069  -4.0093  -1.4705   2.0863  -1.5544
#> 
#> (1,2,.,.) = 
#>    2.1180   4.1878  -9.4852   3.0872  -5.9913
#>   -4.6519  -3.2692  -8.3674   2.1044  -5.9970
#>    4.4171  -2.0075   3.2011  -3.1662  10.3383
#>    0.4309  -2.0497   0.5170   0.2712   5.3997
#>   -2.8957  -4.8481  -4.3199  -7.8721  -8.6972
#> 
#> (1,3,.,.) = 
#>   0.5656 -3.9867  1.0657 -9.6112  5.0062
#>  -0.3649  3.4852  3.4122 -3.5907  5.4748
#>   2.6681  1.2561 -3.8738  3.7531 -7.2294
#>  -0.0736 -1.9148 -5.8057  1.3367 -3.0906
#>  -2.3046  1.7154  4.9894  0.3659  4.7124
#> 
#> (1,4,.,.) = 
#>  -3.6260  5.1872  2.6341 -0.7503  5.0554
#>   3.7981 -3.1711  4.2502  6.3260 -1.9682
#>  -4.7098  2.1837  4.3983 -10.9280 -0.3849
#>   1.8251  4.1376 -1.1273  1.3935  9.8497
#>  -0.6894  1.3434 -1.8925  6.8985  1.5282
#> 
#> (1,5,.,.) = 
#>   2.3366  1.8809  1.1890 -4.8926  1.2610
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
