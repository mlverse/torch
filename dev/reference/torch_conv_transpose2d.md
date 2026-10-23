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
#>   2.6812  -0.8109   3.7026   7.0589  -5.7190
#>   -8.8469  11.1101   2.5970  12.1490  -0.2014
#>   -5.8089   5.3805  -6.1637  -0.9461  -0.4638
#>    9.6843  17.5369  -0.7012  14.1604   1.6379
#>    3.6247   2.0707  -6.7428   2.6172   2.3439
#> 
#> (1,2,.,.) = 
#>  1.1572  5.2673 -2.0206  1.0293  0.2850
#>  -7.8964  5.0376 -3.6870  0.6691 -2.0145
#>  -7.5508 -3.8604  1.4349  0.5239  6.7707
#>   4.8910 -7.0829  3.4068  2.7012 -2.8625
#>   2.1703 -3.9020  1.7758  1.2032  5.1890
#> 
#> (1,3,.,.) = 
#>  1.4897  0.7630 -0.2233  2.4386  5.3545
#>  -1.0515 -5.3533  0.2775 -8.2376 -5.2474
#>   7.2972 -1.0748 -2.0265 -1.2122  2.8032
#>  -4.5465 -2.7206  4.1073 -3.2933 -3.5254
#>   3.2399 -1.0278 -7.3143  5.3531  4.0120
#> 
#> (1,4,.,.) = 
#>   0.2455  -1.8884   3.8361  -2.2059   2.5647
#>    5.5573   3.0080   5.9260   6.8629   2.8024
#>    2.6201  -0.9386 -10.3877  -3.8871  -3.6390
#>    0.7206  16.5628   2.6141   7.9088   5.1761
#>   -5.4823   0.9920  -5.2118   0.8776   0.9020
#> 
#> (1,5,.,.) = 
#> -0.8365  4.9735 -0.7534  5.4881 -3.3674
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
