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
#>  -1.5981   8.7293  -0.5981  -5.2424  -2.6332
#>    7.8195   6.1796  -3.6290 -11.2518  -5.4049
#>   -1.3261   2.9224  -0.0746  -4.1659  -5.6370
#>   -4.0933   6.0488   0.8734  -3.8935  -7.5958
#>    1.4225  -4.4372  -4.2180  -8.2096  -1.6386
#> 
#> (1,2,.,.) = 
#>  -2.2692   2.8494   7.5801   5.0365  -2.2031
#>   -4.8313  -6.2624   3.6672   0.6974   3.1394
#>   -2.1633   0.9194   0.9484   5.1736   0.3023
#>    0.1774  -4.7682  12.1595   2.7958   1.5440
#>    9.6727  -6.4131  -4.2306   4.9929  -2.1443
#> 
#> (1,3,.,.) = 
#>  -2.7133   0.0473  -0.0289   0.6913   3.5722
#>   -0.2355  -8.3871   5.2940   4.2940  -4.8280
#>   -0.4500  -2.8552   6.9310   6.1930   0.1648
#>    0.7044  -1.9189  -0.7560  14.4401  -0.8180
#>    2.8884  -1.9470   9.9847  -5.0215  -7.6201
#> 
#> (1,4,.,.) = 
#>  -1.8206  -2.2700  -5.0676   1.1474   5.9209
#>   12.0527  -0.7313  -7.3126   7.5070   1.8350
#>    8.3420   9.1745  -0.8657  -0.8131   3.1299
#>    5.2401   0.1846  -4.1233  -1.1507   5.0625
#>   -5.3062   5.4694   4.8674  -1.7463   4.1413
#> 
#> (1,5,.,.) = 
#> -4.7379  4.8794 -4.3167  1.8394  2.2897
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
