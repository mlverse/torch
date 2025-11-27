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
#>   6.6944  3.1726 -2.1231 -3.6735 -3.7932
#>   0.0336  4.0342 -6.7299  3.1149  0.6983
#>  -3.4913 -9.6226 -12.5407 -7.2503  5.7891
#>  -6.1179 -5.1970 -5.0721 -1.8941 -8.9733
#>  -1.4283  0.6434 -1.1612  2.9740  3.1161
#> 
#> (1,2,.,.) = 
#>  -1.5368 -4.5693  6.2912 -5.1752  3.1152
#>  -1.4287 -7.8269 -0.8515  7.5172  6.6175
#>   8.2542 -1.6832  2.9444  4.7447 -4.1997
#>   0.3772  6.6211  8.0563  6.3859  4.9564
#>   0.7602  4.8574  6.3156  0.3569 -4.2583
#> 
#> (1,3,.,.) = 
#>    0.3875   2.8632   2.0483  -3.2639   3.4115
#>    5.8548   5.7232  16.2090   1.4865 -10.1078
#>   -1.0903  -0.7657  -3.4366  10.4168   1.6881
#>    8.5976  -3.6152 -13.2137  -9.8532  -8.8492
#>    1.2816   3.2298   2.4003   1.0421  -3.6896
#> 
#> (1,4,.,.) = 
#>   -0.8303   0.2497  -1.4269  10.9454   1.6830
#>    3.5615   2.7059  -8.1235  -1.5430  -2.1136
#>    2.2077   1.4548   6.4514  -7.3830  -3.0035
#>   -7.8301 -12.4495  -5.9826   0.2705  -3.9024
#>   -1.0564  -9.2451  -5.6213   2.0841  -4.5860
#> 
#> (1,5,.,.) = 
#>  -2.7521  0.2387 -2.3690  8.0903 -2.1027
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
