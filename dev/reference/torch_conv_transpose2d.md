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
#>   0.0390   0.7376  -1.0762  -5.8470  -2.3907
#>   -3.5169   2.2841   4.6301  -3.8218   3.1633
#>    5.5405   2.5324   4.8238   3.1803  -4.6966
#>    5.1131   4.1903 -13.5437   5.8987  -8.6034
#>    6.0791 -15.0587  -1.1404  -0.2412   2.7875
#> 
#> (1,2,.,.) = 
#> -0.7925  7.0617 -3.2227  6.3347  1.0885
#>   4.3882  2.9209 -3.1346 -5.9915 -3.4936
#>   5.4081  7.6592 -1.6901 -7.9987  1.5064
#>  -4.5268 -6.0961 -2.2592 -4.8364  2.7143
#>   3.2706  1.7208  9.7851 -2.8791 -1.5834
#> 
#> (1,3,.,.) = 
#>  1.2375 -2.7140 -2.6287 -1.0101 -1.1687
#>  -2.5599  2.7208  3.3664  5.8785  7.9297
#>  -7.3104 -2.6639  0.7841 -2.4115  6.8813
#>   0.9146 -0.1766  4.9495  0.2117  1.1895
#>   1.4618 -4.9326  0.3474  6.6611 -0.4280
#> 
#> (1,4,.,.) = 
#>   2.0075   7.0726  -4.9879 -10.9248  -4.0899
#>   -6.8667   6.1934   8.6746  11.8801   9.1117
#>    5.0560   4.8229   4.0831  -9.7811  -3.9290
#>    8.3410  -0.6996   6.3793   3.5741   1.5175
#>    1.5707   3.6292  -5.2788   2.3024   5.4104
#> 
#> (1,5,.,.) = 
#>  0.1335 -6.5044  2.6535 -0.1726  0.6521
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
