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
#>  2.3271 -3.9975 -1.9203 -7.0533  3.2369
#>  -1.3142 -8.3641  3.8747  3.3001 -3.2694
#>   0.2799 -1.5340 -2.0432 -2.0519 -3.2512
#>   0.8928  6.3915  2.9747  1.4458  1.6235
#>  -0.7944  0.0013  2.2646 -0.5158  0.1982
#> 
#> (1,2,.,.) = 
#>  -4.5015  -0.1483  13.1557   4.1733  -3.2744
#>    2.1231 -12.0932   4.0093   4.8984  -0.1358
#>    1.3740   2.3286 -11.8808   3.6706  -0.3147
#>   10.3639  -8.0875  -5.6795   4.5168   1.3609
#>    0.9841  -6.3750  -5.6370  -1.3826  -0.2205
#> 
#> (1,3,.,.) = 
#> -0.6149 -4.5716  1.5601  3.5789  2.6765
#>   4.2706  3.7016 -6.3354 -0.2712 -0.8458
#>  -4.6352 -6.2134  0.9322  5.7838 -5.2666
#>  -2.4911  4.4623  1.4988 -7.0723 -0.8324
#>   3.2088  3.0706 -9.9945 -2.4778  8.8618
#> 
#> (1,4,.,.) = 
#>  -2.5635  -2.1682  -1.5164   0.7157  -3.4773
#>    3.6547   8.7107  -0.8603 -10.7183   3.6074
#>    0.5635   7.7750   5.2367  -2.9962   8.4931
#>   -4.6769   7.9172  -8.5584  -7.0718   4.9439
#>    2.2299  -3.9599  -1.7581   1.1978   0.0395
#> 
#> (1,5,.,.) = 
#>   3.1969   9.4981   3.4401  -4.5614   3.7781
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
