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
#>   5.0281   3.3648  -2.1794  -4.6189   3.4055
#>    3.1571  -2.4784   0.1688   3.0259   0.4292
#>    3.7742  -1.3088   2.2756   7.5925  12.1211
#>   -5.1398  13.6979  -0.7514  -8.5121   0.4135
#>    0.4580   6.6765  -9.6323  -5.5321   0.3354
#> 
#> (1,2,.,.) = 
#>  -1.3017   5.9593  -3.5260   2.1693   5.4466
#>   -0.7800   3.3742   5.0371  10.0120  -3.4217
#>    3.2387  -3.8259  -1.3568   9.0201  -4.2141
#>    0.0740   5.7854   4.4577  -4.9658   6.6584
#>   -0.3107  -0.6642   7.0061  -2.1604   4.0138
#> 
#> (1,3,.,.) = 
#> -5.4122  1.7675  3.1405  0.7711 -0.7769
#>   3.7387 -4.5457 -7.6755  5.2568 -9.9684
#>   0.9325 -0.4486 -6.8755  7.7950 -1.8563
#>  -6.6833  0.6599  9.6186 -1.2567 -2.6449
#>  -3.9380 -5.4663  5.5581 -4.3951  3.9596
#> 
#> (1,4,.,.) = 
#>  -4.2693  -2.0444  -4.3457   4.9904   4.0844
#>   11.7861   2.5289   0.2558  -4.6685  -2.5838
#>   -1.4030  -5.9612   1.3634  -4.5041   2.2055
#>    2.6659  -2.9960  -3.7859   2.3055   9.5523
#>   -5.0766   6.7231   7.1530 -16.4729  -8.5939
#> 
#> (1,5,.,.) = 
#>   1.3991   1.6689   0.2655  -1.3596   6.8373
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
