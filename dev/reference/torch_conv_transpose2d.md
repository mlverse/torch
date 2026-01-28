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
#>   0.0511 -1.5552 -3.2886  0.4290  3.5621
#>  -0.1275 -5.8962  1.7635 -0.4712  5.9651
#>  -13.1286 -0.0124 -6.2785  2.9698 -4.6212
#>  -2.9704 -0.3921 -0.6063 -8.1091 -3.9509
#>   4.6230  4.1731  5.0612  1.5358  1.4420
#> 
#> (1,2,.,.) = 
#>   -6.7240  -8.4169 -13.0215  -1.6085   3.0339
#>    2.1061   8.7010  13.5349   2.0704   7.4183
#>   -7.0231  -7.9118  -9.7988  -0.5561  -5.8352
#>    8.0471  -2.1559  10.4702  -0.4938  -1.2038
#>   -2.0180  -3.9955  -1.7071  -4.0451   0.6400
#> 
#> (1,3,.,.) = 
#>   -2.1504   0.4074  -4.2120  12.4693  -6.8285
#>   11.6377  11.0029   4.4349  -8.2749 -10.2099
#>    2.3776   1.3237  -1.0748  -7.0285   2.6975
#>   11.3928  -3.9697  -2.9718  -2.8932   3.8453
#>    0.6517  -6.7367   2.2182   3.8212   7.1054
#> 
#> (1,4,.,.) = 
#>  -4.4558 -2.9041 -2.2585 -4.0039  1.5385
#>  -1.2900 -3.3862  1.9002  5.5024  3.2634
#>   2.5409 -1.4138  2.5830 -0.1033 -3.2716
#>   0.3074 -7.7467 -2.1344 -1.9935 -4.5837
#>  -0.7418  6.8679  2.6398  1.4231 -3.8745
#> 
#> (1,5,.,.) = 
#>  -4.5319 -0.9919 -1.5833  4.3561  0.6429
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
