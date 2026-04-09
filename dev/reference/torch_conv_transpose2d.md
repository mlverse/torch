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
#>   2.5920   3.2921  -8.0964   0.1122  -2.6042
#>   -0.0081  -4.8213   5.4066  -3.6309   3.6165
#>    4.0462  -5.9077  -6.3484  -7.8330 -13.5959
#>    1.5035  12.6569  -2.2661   2.0544  -1.8352
#>   -4.1547  -5.4867  -3.2966  -5.1569  -2.5423
#> 
#> (1,2,.,.) = 
#>  -2.1479   0.2155  -0.5863  -4.8144   1.1681
#>    5.8023  -7.1864   3.8861  10.8374   5.2535
#>   -8.3932   5.1784  -4.0829   8.2588   4.2977
#>   -7.6547  -5.7317  -0.5420   2.1930   5.4176
#>   -1.7766  -0.7172   1.1516  -1.5852  -2.0731
#> 
#> (1,3,.,.) = 
#>   0.5614   3.1085   0.4995   1.7291   2.3861
#>   -0.2000  -0.2477   7.9848  -1.0942   0.3112
#>   -9.3330  -3.2305  -6.5993 -10.3351  -5.3692
#>   -6.3659   2.5838   2.3081  -2.8099  -2.8119
#>   -4.0310  -2.1315   0.2516  -3.0107   1.9102
#> 
#> (1,4,.,.) = 
#> -2.1109 -4.4938  3.3292 -0.1819  0.1232
#>   7.2279  6.3636 -6.6163  5.1628  0.4833
#>   0.4047 -1.4119  4.2301  8.1976 -0.7037
#>   1.0898 -6.0144 -6.7031  1.1592 -4.7419
#>  -0.8603 -0.3363  5.9933 -0.1841  3.2223
#> 
#> (1,5,.,.) = 
#>   1.1961  -1.8760  -5.6800   6.6773  -2.7492
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
