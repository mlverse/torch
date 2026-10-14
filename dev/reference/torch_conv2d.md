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
#>  -4.4735   8.8153  -0.0255  -1.1750  -2.7836
#>    3.3782   1.0880   8.1039 -10.0648  -7.9618
#>    3.9435   7.9266   5.6325   3.3898   0.9572
#>   -3.8443  11.1061   6.9657   6.5146  -6.4471
#>    4.5296  -1.8742   2.3131   3.7094   4.2129
#> 
#> (1,2,.,.) = 
#>  0.8463  7.4020 -6.2158 -2.4517 -6.4263
#>   3.2693  5.9601 -7.3469  5.5563 -2.1925
#>   8.2356  3.3716  8.4018 -7.7765  4.2051
#>   4.5645  3.6927 -5.5052  1.0559 -9.6720
#>  -2.3209  4.0877 -0.7040 -5.2029 -4.8642
#> 
#> (1,3,.,.) = 
#>  -2.4259   0.9013   0.3715  -5.5263   0.2774
#>   -4.0273   0.7561   4.2179   6.4354 -16.0790
#>   -2.0789   0.2874  -0.3926  12.7575   2.6410
#>   -0.7497  -0.6089   0.2765  10.9363  -2.3738
#>   -9.7454  -2.0056  -8.0726   2.4384  -0.9707
#> 
#> (1,4,.,.) = 
#>   3.6010  -7.5713  -8.7391  -9.7133  -2.5884
#>   -3.4484  -4.9488  -2.3872   8.9779  13.8685
#>   -2.5548   1.5096  -4.0072 -10.5759   5.5656
#>    3.7736  -4.0822  -5.1169  -3.6001   9.4578
#>   -1.5780   0.1269   1.7116  -4.7138  -6.4357
#> 
#> (1,5,.,.) = 
#>   3.7381  -4.7682  -9.3025  -3.6844  10.5716
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
