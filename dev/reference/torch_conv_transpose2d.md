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
#>   2.0611  4.6484  4.1242 -0.4303 -3.1393
#>   4.4714 -4.1188  6.8083  1.7144  1.0285
#>  -2.0685 -4.7054  4.1368  4.3236 -1.0793
#>  -3.0841  3.6479  0.6700 -2.0269  1.3146
#>  -2.8585  5.4716 -3.8506  4.4836  0.5848
#> 
#> (1,2,.,.) = 
#>  -4.7940  4.8057  2.9744 -0.8114 -1.5732
#>   1.6053 -2.1888 -7.0660 -7.6203 -2.6736
#>  -1.5364 -0.6740 -3.4695  6.0508  1.1116
#>   9.8655 -1.6661  3.9267  6.3783  4.5033
#>   1.9825 -1.3368 -1.8431 -2.9167 -3.5262
#> 
#> (1,3,.,.) = 
#>    3.2697   2.5046   3.3615   5.4187  -3.6926
#>   -4.0839  -6.0124  11.0849   4.2152   3.4250
#>   -5.1888   1.0908   2.4143   3.1120  -3.1736
#>    4.2121  -4.5720  -4.7707   2.6270  -4.1060
#>    1.4449  -7.7279  -4.1002  -3.3175   3.2491
#> 
#> (1,4,.,.) = 
#>    3.9315   8.1568   0.8782   3.4485   4.4622
#>    2.0469  10.4318  -1.6500   3.4662  -3.8935
#>   -0.3278   1.1837   1.7729  -6.9014   1.1101
#>   -4.7717  -2.4310   3.9116   0.5673   3.2060
#>    0.8977  -1.0159  -8.9416   1.9858   1.2152
#> 
#> (1,5,.,.) = 
#>    3.8661   2.3795  -4.7510  -5.4598   5.1479
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
