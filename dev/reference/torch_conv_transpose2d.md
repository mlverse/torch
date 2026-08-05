# Conv_transpose2d

Conv_transpose2d

## Usage

``` r
torch_conv_transpose2d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padH, padW)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padH, out_padW)`.
  Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

## conv_transpose2d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
inputs = torch_randn(c(1, 4, 5, 5))
weights = torch_randn(c(4, 8, 3, 3))
nnf_conv_transpose2d(inputs, weights, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#>  -1.6616   0.0137  -6.1094  12.0189  -7.3687
#>    0.3018  -9.9261   5.7359   6.2925  -2.1599
#>   -6.8296   1.7846   0.4840  -0.1206  -1.0212
#>   -2.5626   2.5035  -2.3364   9.7845  -0.1807
#>   -4.9740   5.9314   1.8926   8.6423   0.5023
#> 
#> (1,2,.,.) = 
#>  -4.3267  -1.7866   1.5270  -3.7930  -2.4699
#>    7.9945  12.3674  -4.0553   5.6058   5.4137
#>   -1.1894  -9.2267  -3.7780   1.2540  -3.2867
#>    1.4207   4.9865  10.5490  -9.0030  -2.3086
#>   -1.1242  -2.8026  -3.6222  -0.3440  -0.8669
#> 
#> (1,3,.,.) = 
#>  -6.6568 -10.0631   4.5932   5.8927  -7.2034
#>    0.5252   7.9606  15.5193  -1.3555   9.6987
#>   -1.4041  -5.8996 -12.1034   3.6059  -0.2265
#>    6.3976  -6.2412   6.2445   5.7989   1.1949
#>    7.0675  -9.4311  -3.7631  -4.7934  -5.4831
#> 
#> (1,4,.,.) = 
#>  1.3087 -1.8244 -0.5475 -5.6405 -1.7390
#>   4.1300  0.9511 -3.7680 -8.6174  8.8853
#>   2.2635  4.2231 -8.2424 -5.2816  0.5352
#>  -1.5589 -1.6229  2.5225 -1.5586 -1.9212
#>  -0.8739 -1.6558 -2.4575 -4.6741  4.3333
#> 
#> (1,5,.,.) = 
#>  -8.7603   2.0489  -2.3902   4.7933  -1.4914
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
