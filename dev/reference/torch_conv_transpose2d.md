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
#>  12.0211   0.1814  -0.6950   6.1835  -0.5045
#>    3.8409  -0.0472  -4.5004   2.2052  -1.0791
#>   -5.2513   0.1747   8.1506  -6.2165  -0.9558
#>    5.8137  -1.4350   1.0964  -0.3347  -2.9705
#>    0.4787   1.3067  -1.6176  -3.9152   8.6978
#> 
#> (1,2,.,.) = 
#>  -6.7153   0.8403   1.5654   0.7454   3.3420
#>   -0.0796  -3.8562   3.7060  11.9828   7.6861
#>    1.7979  -1.9988  20.9150  -0.7449  -8.2982
#>  -11.5979   0.5937   9.2977  -3.1341  -0.0032
#>   -6.1905  13.9734  -3.7389  -1.5388  -7.7772
#> 
#> (1,3,.,.) = 
#>   0.1526  -1.1648  -1.9877  10.2701   1.9763
#>    8.0053  -5.7082   6.7503   1.5603  -3.2026
#>  -10.3681  -2.4134   3.7766   0.8478  -1.3618
#>    0.1189  -3.5456  -0.9279   4.6514  -4.2598
#>   14.5685  -4.5960  -0.5132   4.9343   3.4646
#> 
#> (1,4,.,.) = 
#>   4.8591  -2.6583  -5.2600   0.7745   5.7788
#>   -5.1925  -3.3448  -0.9899  -2.4930   1.3425
#>   -3.7210  -9.3307  -4.9581  -3.7057  -3.7873
#>   -3.3203 -12.7546  -2.6357 -11.0112   0.1672
#>    0.0751  -6.2648  -0.2368  10.6847   4.0135
#> 
#> (1,5,.,.) = 
#>   4.7978  -5.1423  -2.3460  -1.3248   2.6922
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
