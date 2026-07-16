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
#>   0.2137   3.7873  -8.1825  -3.4765  -2.5253
#>    3.2374  -1.9393  -3.3461  -6.7727  -6.2712
#>    6.4501   2.0420  -0.2593   5.2239  -7.2747
#>   12.5788  -3.3023  -5.0253   2.0471  -0.2311
#>   -7.7190  -4.7219  -4.4835   2.0755   6.6828
#> 
#> (1,2,.,.) = 
#>   2.3441   5.1210  -5.2434   2.8181   2.7385
#>   12.7721   6.5361  -5.2549  -8.1933   2.3876
#>    4.0514  -4.6258  -4.4757  -5.3530  10.0553
#>   -0.1125   1.1600   0.3901 -10.2580   9.5472
#>    4.3693   3.5331  -4.1013  -4.3446  -4.0106
#> 
#> (1,3,.,.) = 
#> -0.9593  4.6119  6.6551  1.1491  3.6175
#>   1.4164 -3.0362  0.7042  7.1041  1.9101
#>  -3.0004  0.7980 -2.8404 -5.9691  9.7032
#>  -2.2516  0.8771 -2.7714 -1.9558 -7.5549
#>   2.0655 -1.6513  1.1569 -7.0452  0.5617
#> 
#> (1,4,.,.) = 
#>   3.8521   0.8321  -5.4776  -0.6987   1.9899
#>    0.4922   7.7034   2.7757  -7.3384 -11.8785
#>   -0.0238  -0.4761   8.7491  -6.8703  -2.5837
#>   -0.8075   0.2597   3.6966   2.3416  -7.5089
#>    6.3739  -1.9343  -4.9322 -11.5228   3.1741
#> 
#> (1,5,.,.) = 
#>  2.2892  6.0792 -0.9607 -4.6541  1.7023
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
