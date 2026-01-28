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
#>   -6.8191   7.0431  -1.6484   2.5281   2.3802
#>   -7.2551  -4.1569   2.0958  -1.0602  -1.0014
#>    6.7657  -8.5446  -1.6197   7.3819  -2.9292
#>    1.0752   1.1675  -7.7041  10.7754  -7.5064
#>    2.3123   5.7489  -2.1185  -7.8203   3.3949
#> 
#> (1,2,.,.) = 
#>  -0.4006  7.2999  8.5114 -6.1332 -5.3709
#>  -2.5383 -5.1312 -5.2550 -8.1776  3.8928
#>  -5.1818 -4.3870  7.6418  1.7328 -5.9701
#>   3.1668 -5.0632  0.6518 -14.6135  3.0157
#>  -2.6670  2.8927 -2.4869  0.2641  6.9734
#> 
#> (1,3,.,.) = 
#>    0.2800   6.6135  -0.7185  12.8447 -10.2305
#>    3.4374  -8.1859   4.9878  -6.3367  -3.0993
#>   -0.0740  -3.4147  -3.7074  11.9760  -1.4796
#>   -7.5408  10.6012 -12.4018  -9.7768  -2.7015
#>    2.8662  -4.0396   7.4771  -1.8102   4.4152
#> 
#> (1,4,.,.) = 
#>    0.6555  -8.5391   0.1079  -6.8594   5.4093
#>    6.3477  12.5705   2.9306   1.2022   3.9559
#>   -1.4276  11.2291   6.8323  -7.7912  -6.5044
#>   -1.0368   5.5935  11.0100  -4.6122  -4.8536
#>   -4.3231  -2.0561   1.2970   3.6638  -4.6668
#> 
#> (1,5,.,.) = 
#>    4.7044   3.3136  -2.5831  -6.3800   3.6129
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
