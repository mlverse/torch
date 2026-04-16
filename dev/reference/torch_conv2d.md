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
#>  -7.3273   1.0250   1.7363  -4.7126   0.5164
#>  -12.0060  -2.1512  -7.1557   2.5119  -0.6484
#>   -6.1028  18.7118  14.3070   1.4125  -0.4248
#>    6.8882   5.5975   2.3562  -6.5382   0.0016
#>   -5.1735  -3.2206  -2.8616 -14.9833  -3.4940
#> 
#> (1,2,.,.) = 
#>   2.2732  -3.3671   4.4606  -0.9941   6.5727
#>   -9.5464  -0.1472   1.6576   5.0093   1.0862
#>   -3.1779   0.5981  14.2865  -7.9691  -1.3621
#>   15.5599   1.9500  -0.1239   1.7466  -4.0737
#>    1.3399  -4.7933   1.7150  -8.0214  -5.4758
#> 
#> (1,3,.,.) = 
#>  3.5147 -5.6607  1.2191  6.4575  6.9704
#>  -6.2195  8.9213  9.0162  0.1503  3.9632
#>  -3.5955 -2.7438 -7.7003 -5.8099 -7.5555
#>   4.7307  1.5080  3.8938  0.4007 -5.5732
#>   0.8117 -1.8080  4.8396 -4.4928 -3.6255
#> 
#> (1,4,.,.) = 
#>  -5.2283  -2.1635  -5.8986  -1.5114  -4.7316
#>   -2.8290   5.1157   5.7656  -3.7021   2.5279
#>   12.3879  -1.9787   3.5445   2.7005   5.9558
#>   -3.2805   1.2238  -7.6901  -4.5366  -3.5980
#>   -0.2078  -2.1291  -9.2439  -6.6530  -0.4182
#> 
#> (1,5,.,.) = 
#>  -0.6139   1.9919   5.6437   9.6972   3.9061
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
