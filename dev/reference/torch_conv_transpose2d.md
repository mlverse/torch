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
#> -1.2713 -1.1822  7.0718  4.1307 -2.0555
#>   2.8431 -3.0598 -1.0745  5.0641 -1.8737
#>  -7.6799  2.3991 -2.4192  1.7593 -2.0912
#>  -4.0159 -1.8370 -9.3558 -1.2963  6.5369
#>   5.1899  4.9189  3.3008 -3.8527  5.8579
#> 
#> (1,2,.,.) = 
#>  -2.7633  -5.1930  -4.4204  -2.8356  -4.3352
#>   -1.0218   2.9443   1.4859   1.0535  -1.6723
#>   -2.2501 -15.8801  -1.4385   9.0594  -3.1356
#>    2.4240  -8.1578   0.5168   3.7346  -3.6880
#>   -1.5315  -2.1452  -2.9332  -4.3519   1.1422
#> 
#> (1,3,.,.) = 
#>  3.3522 -2.7155  1.5923  0.3667 -1.5824
#>   5.6991  3.4407 -6.0606  3.4131  2.9810
#>  -8.0569  4.0197 -7.7218  3.4687 -0.9831
#>   3.7728  1.7394  0.7564 -2.6649 -7.5751
#>   6.7948 -3.7852  9.5693  3.6688  3.4854
#> 
#> (1,4,.,.) = 
#>   2.4389   2.5857  -5.8364  -2.9577   0.1187
#>   -2.3316   1.6993  -2.7062 -12.7511   0.9515
#>   -0.6569   0.7281   3.1556   1.1701  -5.2711
#>   -7.1719  -0.4800  -1.5196   1.7936   4.5954
#>   -3.5161  -3.7355  -6.3863  -6.0232  -3.8332
#> 
#> (1,5,.,.) = 
#>  4.6107  1.9367  2.5092  2.4783  0.9141
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
