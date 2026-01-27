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
#>   -0.7028  -0.8192   1.0851   2.2998  -3.8485
#>    2.9821  -0.0747  -3.8819   0.8071  -9.0522
#>    0.1999  11.6687   2.8225  13.3537  -4.4198
#>   -8.4078   4.0899 -13.6429  -6.0207  -3.3492
#>   -2.4980   2.5255  -2.5701   0.3913   1.4785
#> 
#> (1,2,.,.) = 
#>    2.0787   5.6386   2.3255  -2.0852  -6.7013
#>   -0.7322  12.7107  -5.1432   1.4219   1.9245
#>    2.3328   0.7612   2.9378  -5.6100  -0.5047
#>   -4.1995  -1.0990   8.2641 -14.8206   1.6612
#>    2.3564  -9.3540  -1.4511   5.0758  -4.5607
#> 
#> (1,3,.,.) = 
#>    6.2200  -7.7194   2.6899  -9.8798   0.0266
#>    2.8047   2.9419   1.2706  10.1525   3.8393
#>    1.8341  -5.4482  -5.9524  -4.7571   0.4638
#>   -6.1774   3.1723   2.0652   2.1727  -4.0093
#>   -2.7800  -4.5857   1.6227  -2.1747  -4.4893
#> 
#> (1,4,.,.) = 
#>   2.9265e+00 -6.9736e+00 -1.0536e+00 -4.9767e+00 -2.0268e+00
#>   9.6798e-04  7.1570e-01  5.4377e+00  1.0651e+00  3.5683e+00
#>  -1.9750e+00  3.1702e-01  2.7015e+00  1.7113e+01  3.0798e-01
#>   2.6294e+00 -1.4820e+00 -2.1954e+00  3.2163e+00  9.1270e+00
#>  -3.0244e+00  6.4790e+00  3.6700e+00  3.4772e+00  3.4004e+00
#> 
#> (1,5,.,.) = 
#>   0.8492 -0.8435  1.1434 -2.1362 -6.4804
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
