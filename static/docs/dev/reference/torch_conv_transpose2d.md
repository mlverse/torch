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
#>   0.1810  2.9957  3.4899  3.0716  5.0040
#>   0.8239 -10.6694  3.0891 -5.0050 -0.0420
#>  -3.7163 -7.2520 -4.9961 -8.9231  7.6545
#>  -3.9635 -10.4815 -6.9768 -5.4135  2.1951
#>  -2.1771 -0.1603 -6.0652  2.5439 -4.9821
#> 
#> (1,2,.,.) = 
#>  -4.6035 -0.4003  7.0373 -3.9030  7.8365
#>  -4.6291  0.8971  3.5380 -0.1026 -4.0470
#>  -0.7980 -18.3321 -7.0752 -3.6331 -3.5998
#>  -0.4348 -5.5739 -6.7516 -5.9265  0.0202
#>  -1.3469 -6.8557 -1.8904 -0.6835 -3.0381
#> 
#> (1,3,.,.) = 
#>  -3.3081 -0.8981  1.8633 -11.8485 -0.1668
#>   0.4828  0.2408 -1.4152 -9.8986 -11.9811
#>   1.9487 -6.1534 -11.5657  8.0979 -1.3263
#>  -2.6160  4.5178  0.8707 -0.5026 -4.2115
#>  -3.2666 -0.7475 -5.5481 -0.4991  2.7601
#> 
#> (1,4,.,.) = 
#>  -0.5566  4.9705  2.1547  0.7278 -2.2802
#>   6.6323 -0.8395 -6.1720 -7.7023 -5.7041
#>  -1.9438 -4.0736 -1.2513 -3.6417  5.7671
#>  -1.4896  7.7106  3.5592 -0.1913  5.2220
#>  -1.9941  0.7876  4.7974  7.4200 -5.5258
#> 
#> (1,5,.,.) = 
#>   4.6151  5.7551 -5.7899  0.4718 -1.2533
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
