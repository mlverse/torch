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
#>   1.1059  -4.9143  -4.3392   2.0309  -4.8082
#>    6.1189   8.0095   7.5386   8.5620   1.2675
#>    1.2880 -13.1824   1.7445  -4.0543  11.3287
#>    3.7612   7.7440  21.0221   3.2357  -0.8205
#>    1.9131  -0.2136   3.3259   4.9855  -4.2251
#> 
#> (1,2,.,.) = 
#>  -4.7765   3.8606  -1.8119   5.6071   0.7851
#>   -2.9135  -2.7948  -9.4800 -12.5971   1.3596
#>   -0.7855  -0.7424   6.4536  -5.1508   0.8596
#>   -3.7242  -3.7920   7.1348   0.2905   1.4432
#>    7.1356  -1.8457   1.2054   2.4416   7.8052
#> 
#> (1,3,.,.) = 
#> -0.4723 -0.9737 -6.7705  2.0056 -3.6145
#>  -0.6183  8.3021 -4.7691  4.8025  1.8162
#>  -4.1534  6.4935 -0.6823 -4.9502 -0.7975
#>   5.8980 -7.8911 -2.3367  8.6197 -0.6945
#>   2.0098 -4.5685  7.2587  6.8020  1.9812
#> 
#> (1,4,.,.) = 
#>   5.0168   1.8588 -14.0417   8.9401   0.6727
#>  -10.7440  -5.2457   5.8567  -7.1885   7.2299
#>    2.7349   7.9988   3.3405  -6.2178   6.3988
#>   -4.8316  -0.1454   5.8467  -0.4603  -3.9255
#>    1.9669   5.0902   4.1483  -1.0841  11.8347
#> 
#> (1,5,.,.) = 
#>  -3.1489   6.1160   0.6345  -0.3743   1.3087
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
