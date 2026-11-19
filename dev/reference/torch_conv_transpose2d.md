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
#>   1.4351   2.0844   4.6751  -1.2152   3.2973
#>   -4.7664   8.2833   0.1862  -1.4199  -2.3487
#>   -5.2914  -3.0285  -3.1999  -4.6419  -4.2063
#>    7.8723  -9.6362   4.3544  -0.6454  10.3991
#>    3.6671  -1.6681  -1.6095  -4.0115   7.8088
#> 
#> (1,2,.,.) = 
#>   2.5405  -4.9814  -4.3440   2.2387   6.7868
#>   -1.5162   0.9378  -8.4669   6.4805   2.6205
#>    4.2267   3.1627  -0.7190 -11.0154   4.7437
#>    8.9131   2.9215   9.5081   5.9554  10.1295
#>   -1.8923   3.1026  -4.8653   4.5445  -4.2970
#> 
#> (1,3,.,.) = 
#>   2.9787 -10.2446   0.7601  -1.6726   5.6050
#>   -6.6724   3.2859   6.4304   8.2446  -4.0352
#>    2.8512   0.9754  11.1231   1.1347   7.2807
#>    0.5748 -12.0839   5.8455   4.4458   3.2316
#>   -2.6021   4.8391  -8.1411  -0.3639  -5.9460
#> 
#> (1,4,.,.) = 
#>  1.3218  5.0709 -0.7804 -2.1834  3.0519
#>   0.5690  2.4216  2.4286 -3.8195  2.4787
#>  -3.1874 -4.1899  0.9659 -8.8006 -0.4841
#>   7.2450 -1.8893  1.3504 -2.1684  0.3157
#>   2.2942  0.5259  7.3624  7.1164  2.6558
#> 
#> (1,5,.,.) = 
#>  -0.2079   1.5187   2.1688  -1.1165  -0.5445
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
