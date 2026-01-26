# Conv2d

Conv2d

## Usage

``` r
torch_conv2d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias tensor of shape \\(\mbox{out\\channels})\\. Default:
  `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a tuple `(padH, padW)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv2d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

See
[`nn_conv2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
filters = torch_randn(c(8,4,3,3))
inputs = torch_randn(c(1,4,5,5))
nnf_conv2d(inputs, filters, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#>   2.3605  4.9064 -5.0987 -2.4492  6.4652
#>   0.3736 -2.1239  1.2090 -0.7651  2.0137
#>  -1.7852 -6.4250  2.6692 -5.4817  2.0674
#>   1.8267 -1.3942  2.3497 -0.6934 -0.1522
#>   0.2457  2.6922  0.9435 -8.8036 -1.8362
#> 
#> (1,2,.,.) = 
#>   2.1740 -7.2545  6.1915  1.3090  2.0514
#>   8.8308  1.2296 -2.3043 -3.7957 -1.5367
#>   3.1908  7.8375 -5.5248  0.3974  1.5165
#>   0.6243 -1.8780 -1.2799  5.4276  0.9992
#>  -0.5053  0.6827 -4.3698 -3.9070 -1.7782
#> 
#> (1,3,.,.) = 
#>  -0.5993 -5.8944 -5.1744 -6.0728 -2.0646
#>   2.6400  0.2730 -4.7275 -15.5702 -2.4643
#>  -4.7877  9.4295 -0.0488 -14.3521  2.4415
#>  -3.7049  3.3142 -2.9583 -1.1658  4.6019
#>   3.7406  1.1570  0.6285  7.9401 -2.7323
#> 
#> (1,4,.,.) = 
#>    0.7021   0.6601   3.6264  -0.4367  -6.3483
#>    9.0191   2.4074  10.7203   3.4570  -2.7378
#>   -3.7791   2.8956   8.6025   5.1466   0.9204
#>   -1.2608  -0.1012  -1.5554   4.6002   2.3740
#>    1.4605  -0.2794  -4.2979  -7.2336  -3.4071
#> 
#> (1,5,.,.) = 
#>    0.4291 -10.8112  -3.3988  -3.4867   2.8276
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
