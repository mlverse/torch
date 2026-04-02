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
#>   1.3254 -2.9182 -2.6645  2.3678  1.7617
#>  -5.9915 -2.1483  2.3870 -2.1254 -3.1892
#>  -8.8511  3.7803  3.7742 -7.6355  2.8798
#>  -4.6848  1.0244 -2.6553 -0.8979 -8.0094
#>  -2.5171  1.1917  1.6256 -0.9869  4.8563
#> 
#> (1,2,.,.) = 
#>   2.2647  0.0517  0.9567 -1.2654  1.0595
#>  -1.7964  2.6257  2.8492 -3.9883  2.7769
#>  -0.6559  0.4571  5.9551 -4.0574  0.7113
#>  -3.2311 -1.0672  8.3449  2.3401 -2.8157
#>  -0.8339 -2.5599  1.8648  1.1960  2.0142
#> 
#> (1,3,.,.) = 
#>   -2.9032   0.0208   2.9989   0.6245   5.4981
#>   -0.1945  -6.3741  -2.7096  -4.5767  -0.5534
#>   -3.0121   2.9377  -0.0440   0.7585  -7.9551
#>   -4.1852   2.4289   7.6763  -7.7448   2.0276
#>   -8.3763  10.8051  -0.6962  -4.8126   4.2862
#> 
#> (1,4,.,.) = 
#>   0.2218 -2.5502  0.4717 -6.4144 -1.6697
#>   1.7375 -4.5465 -4.8161 -6.7923 -3.0683
#>  -2.6763  2.7025  2.1086 -1.9302 -1.8666
#>   4.4962  2.4307  1.8831  3.6146 -0.1600
#>   1.3528 -2.6165  7.5410 -2.8936 -3.1382
#> 
#> (1,5,.,.) = 
#>  -7.4201 -4.8205  4.8065  1.9017  4.9397
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
