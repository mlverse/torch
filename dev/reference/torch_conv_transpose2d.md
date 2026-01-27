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
#>    1.9639  -1.4671   2.1423   3.2027  -0.1090
#>   -3.1678   4.9454   7.6043   1.2411  -1.7998
#>   -0.1717  -2.1079  -2.3815   5.9498  -6.8942
#>   -1.0231   4.2592   7.5534  14.9566  -0.8284
#>   12.1912  -6.4154  -9.4397  -0.5560  -4.9090
#> 
#> (1,2,.,.) = 
#>   1.1541 -5.2005 -3.8615 -3.6908  1.2316
#>  -7.3264 -3.3231 -11.0154  4.4964  4.5014
#>  -5.9431  1.6723 -12.6382  2.5019  1.9458
#>   3.3877  3.1922  0.6643 -8.0537 -5.3983
#>   7.6971  0.9810  0.9992  4.0381 -1.3563
#> 
#> (1,3,.,.) = 
#>  -3.0519 -3.3934  4.3215 -1.8298  0.8341
#>  -8.8636  0.0635 -10.7307 -3.0406  1.1844
#>  -4.6801 -1.5044 -8.3504 -1.2539  0.5196
#>  -7.9605  3.0227 -10.5294  0.4238  2.7508
#>  -1.5319  0.2250 -1.8828  3.9005 -1.6975
#> 
#> (1,4,.,.) = 
#>   -3.7022   4.6355   0.6423  -3.3241  -6.2502
#>   -4.9429  -2.8610  10.9130  -4.1425  -1.1035
#>    0.0360  -1.6953  -2.8783  11.0398  -1.8903
#>   -0.7542  -0.8425   2.0190   0.4176  -7.7864
#>   -0.9050  -4.0713  -3.5661   8.9520  -3.3813
#> 
#> (1,5,.,.) = 
#>    3.2825   6.2364  -3.5956   4.7152  -4.1940
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
