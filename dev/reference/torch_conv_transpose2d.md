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
#> -1.4636  3.5161 -2.7510 -2.8869 -0.4838
#>   0.8258  5.1777 -8.2110  2.8304 -0.0019
#>  -1.5254 -3.8930 -5.1843 -6.7888  3.2270
#>   1.3018  2.4710 -0.1588  8.3189  5.5026
#>  -0.7152  0.1423  0.2256 -1.8368  2.3311
#> 
#> (1,2,.,.) = 
#>   4.1432   5.9676   0.5320   2.5029  -3.7740
#>   -1.3973  -0.5129   2.7479  -4.4615  -0.5974
#>   -7.2364  -5.1928 -11.0599   2.4220  -2.9429
#>   -3.4015 -14.7040  -6.2951  -2.1509   1.9241
#>   -3.1988   0.5072  -2.5737   5.4147   1.0831
#> 
#> (1,3,.,.) = 
#>   6.5032   1.3897   5.4583   4.0585   0.9590
#>   -9.3080 -13.5038   1.2196  -1.2739   0.6051
#>    4.0024   8.9190  -2.1227   8.5741  -0.4870
#>   -1.7317   5.6270   6.8652   1.0116  -4.0103
#>    0.2482   2.8695   6.1423   6.6906  -4.3282
#> 
#> (1,4,.,.) = 
#>  -4.5766   5.4295   6.0979   3.6568  -1.2714
#>   -4.2877 -11.1995  -4.1214   4.5560  -0.6576
#>    6.3589  -3.2076  -3.6959  -4.5781   1.6753
#>  -11.9048  -2.1476  -1.5277   6.2448  -3.4997
#>    4.4947  -8.2989   6.8716   4.8715   0.2789
#> 
#> (1,5,.,.) = 
#>   0.0397   0.1034   2.5267   0.8907  -0.9782
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
