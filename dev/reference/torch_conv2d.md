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
#>   3.0142   7.9718  -6.3867   0.1280   0.1532
#>    0.7884  -7.6463   1.8996   0.2458   0.9036
#>    8.0240  -1.9302   3.6131  -1.1685  13.0982
#>   -2.9785  -1.4555   8.1179  10.9086   0.7197
#>   -2.8127   6.5867   4.4043  -4.5922  -4.9294
#> 
#> (1,2,.,.) = 
#> -11.2125  -3.9892   3.7212   2.6765  -0.5384
#>    1.1891  -4.8269  -8.1499   1.7570   3.0769
#>   -4.2480   0.7344  -2.1864  -5.9013   2.0450
#>   -4.8539   2.4675   3.1776  -7.0708   0.1984
#>    3.2302  -2.7235  -6.6607   6.9023   6.2109
#> 
#> (1,3,.,.) = 
#>  -2.6936  -3.2799   6.2219   3.2571  -5.6118
#>    2.3050  -7.3835   1.1039   1.7409   0.4302
#>    1.3502  -1.7792  14.9399   6.4146   3.3169
#>   -3.2896  14.7399  -1.1434  12.3169   6.3206
#>    1.6005   1.2003  -5.4365  12.9446  -1.7810
#> 
#> (1,4,.,.) = 
#>  -0.2322  -1.2839  -2.6034   2.0354   1.9681
#>    3.2703   7.1126  -8.5117  -8.7136   4.4626
#>   -4.8351  14.2489   1.7932  -6.4518  -2.5356
#>    0.7147  -9.8909   1.5154  10.4884   0.8177
#>    0.3864   5.0764   2.0024  -0.4506  -0.7291
#> 
#> (1,5,.,.) = 
#>  -4.1491   3.4691  -4.4991  -2.1291  -5.8537
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
