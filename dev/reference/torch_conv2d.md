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
#>   4.6651  -1.1196  -3.4379   1.0707  -6.8942
#>    4.9546   3.8082   8.9779  -3.5466  -4.5558
#>   -4.2550  15.3808   0.2215   0.9210  -7.4651
#>    0.6135   3.2027   0.9626  -1.2774  -2.1881
#>    3.8768   1.5572   5.0449   0.0633   4.8512
#> 
#> (1,2,.,.) = 
#>   7.3644  -9.0711   4.8841   4.3043   1.6028
#>    0.5116  -3.5799  -6.7840  -5.2706   4.1797
#>    3.5317 -16.2378  13.2992   5.0904  -8.6279
#>    2.7786   4.6746  -2.9173   9.7115  -4.9698
#>    0.6505  -1.1273  -7.3022  -3.0929  -6.5184
#> 
#> (1,3,.,.) = 
#>  0.7825 -2.7258  2.3546 -0.3551  0.6120
#>   0.0225 -4.0027 -0.3949 -1.2883 -5.1061
#>   4.2129 -9.0165  4.3112 -1.7354 -4.1724
#>   1.4359 -4.2003  1.9834 -0.0383 -3.7369
#>  -3.3323  2.9516 -3.4575 -2.6005 -7.3555
#> 
#> (1,4,.,.) = 
#>   3.9238 -10.4197  -3.4615   2.2520   1.6998
#>   -5.7134   0.8638  14.2258  14.8705   1.1120
#>   -0.8507   5.6112   1.6557   2.5442 -11.7708
#>  -15.4754  -7.8109 -16.0634  -8.0434  -9.7099
#>    0.6392   6.1193   6.2876   0.0057   0.8030
#> 
#> (1,5,.,.) = 
#>  -8.0204   2.1951  -2.6619  -0.5978   3.5765
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
