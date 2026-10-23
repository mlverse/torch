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
#>  -1.3982  -0.9060 -10.5765  -7.5329   6.1151
#>   -0.0203  -5.2618  -0.6263   8.1468 -14.8974
#>   -6.0209   3.9994   1.8821  -8.2988  -1.4604
#>    1.1421  -2.6401   0.0214  12.2228   6.4217
#>   -1.4447   6.7592  12.7704  -5.4404  -1.8733
#> 
#> (1,2,.,.) = 
#>   6.6720  -3.0780   9.0745  -4.1291   0.0156
#>   -2.7431  -2.2333   2.7795  -0.8784   8.8737
#>   -3.0314   3.4203  -0.2256  -0.3216   2.5041
#>    6.6450   7.6913 -10.7124  -8.3517 -11.5535
#>   -3.6202   2.3428  14.8446   0.1412   6.4896
#> 
#> (1,3,.,.) = 
#>  1.8928 -6.6405 -0.2735 -5.8846  6.1115
#>  -0.0675 -7.2680 -5.1636 -0.6535 -1.7352
#>   0.3637 -1.0793  2.5477 -5.6009 -9.1674
#>   6.8745  3.8943 -2.4466 -4.2596  7.1993
#>  -3.7198  1.7241  7.0981 -3.0648 -2.9164
#> 
#> (1,4,.,.) = 
#>  -0.0469   2.4307  -2.3285  -3.4157  -0.0622
#>    0.8164  -3.1724   3.1840   6.6028  -2.0084
#>   -3.5021  -0.1680  -1.0718 -10.4946   3.8390
#>    2.1205   0.0245  -8.2912   9.2447   2.9168
#>    0.4444   1.2340   4.5472  -4.7084   3.2072
#> 
#> (1,5,.,.) = 
#>  -0.5244   7.8710  -0.1670  -2.2237  -0.8389
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
