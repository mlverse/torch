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
#>    0.5187   8.7808   0.2380  -3.1715   6.0965
#>    1.5445   0.4269   7.3495  -6.5487   3.9660
#>    1.1047   0.7861  -3.4300  -2.5148  -0.9224
#>   -1.5786  -8.7180  -1.6944  11.9903  -2.5710
#>    5.4422   2.6503   0.0985  -6.9978  -6.1177
#> 
#> (1,2,.,.) = 
#>   -0.3145  -2.4701  -6.0648   5.4990  -5.0745
#>    1.4698  -6.4335   5.9515   6.2977  -3.9847
#>   -1.4729   6.9429  10.7682  -8.8683   2.8818
#>   -2.0104   1.4522  -0.8214  -7.6475   6.6074
#>   -0.3519   2.2498  -3.8058  -1.0542  -0.7669
#> 
#> (1,3,.,.) = 
#>  -0.8106 -7.9398  3.5543  3.4854 -3.4460
#>   2.9980 -1.8128  2.9892  1.1770  1.2185
#>   0.4800  2.7531 -10.2211  3.9638  0.1020
#>   0.5236  0.0146  2.4300 -1.9121 -5.0888
#>  -3.5954  7.8659  9.9378 -1.5992 -6.2481
#> 
#> (1,4,.,.) = 
#>    1.1116   3.3187   9.8329  -0.5803   1.6513
#>   -7.6922  -0.2669   4.0831 -12.4357  -0.5165
#>    1.0909   4.9840  -2.4484   0.8426   5.9088
#>    2.8367   6.5986   4.8975  -8.2787  -4.2553
#>   -2.9535  11.7101   6.3385  -1.6897  -2.8092
#> 
#> (1,5,.,.) = 
#>   2.6973 -2.7978 -1.3529  1.0565  5.8367
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
