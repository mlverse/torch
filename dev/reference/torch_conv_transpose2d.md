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
#>   1.2743  -5.9267 -10.4415  -1.1052   0.5899
#>    3.7445  -6.3630  -4.4571   2.7755   0.6382
#>    3.3016   3.8629  -8.3158   1.9376   4.1398
#>   -0.6872  -5.3722   1.5053  -5.6409  -5.2443
#>    3.6599   3.8068   0.3659  -1.3971  -1.2749
#> 
#> (1,2,.,.) = 
#>  -0.7015  -6.1096  -0.2546   5.2093  -5.8657
#>    5.7254  -1.3452   3.7186  13.0857  -5.3111
#>   -8.4992  -7.9698  -0.1518   5.1404   0.8184
#>    1.1911   0.2341   0.3634  -3.2256   0.3161
#>    4.4105   5.9123  -0.9221   2.4962  -1.9859
#> 
#> (1,3,.,.) = 
#>  -1.7986  -3.8235  13.7339   5.8588  -3.9935
#>    3.9653   3.8229   0.8707  -0.7753   1.7560
#>   -1.3101  -3.5106  -2.3230   0.1714   6.8477
#>    3.5756   0.5353   6.5523   5.3101   5.4899
#>   -2.6690   2.0003  -9.2965   6.7241   5.2113
#> 
#> (1,4,.,.) = 
#> -7.2666 -5.3849  7.7778  5.4260  3.4405
#>   4.1611 -8.6782 -4.0419 -1.6735  4.2885
#>  -2.4320 -3.7897 -0.5329  2.6155 -0.3990
#>   4.7287  6.7425  4.2849  1.3066  2.1291
#>  -1.2397 -5.0961 -5.1118 -3.6258 -5.0782
#> 
#> (1,5,.,.) = 
#>  -4.6179   1.9441  -2.9854   4.2460   1.0841
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
