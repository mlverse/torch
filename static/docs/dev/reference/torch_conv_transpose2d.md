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
#>  -10.2741  11.3147 -11.7798 -11.3927   1.1381
#>   -2.0294  18.7297  -1.9878   1.6134  -1.8279
#>   -0.9779   6.9258  -5.5934  19.3604  -3.2107
#>    4.7684   0.9183  -1.0915  12.2079 -10.4649
#>    0.7360   6.4830  -2.3308   8.1745  -5.2316
#> 
#> (1,2,.,.) = 
#>   1.2170  1.0844 -3.6980  1.9829  2.8397
#>  -2.1488 -8.4568  2.2463  1.9687  1.2951
#>  -4.3752 -10.4047 -0.3380 -2.8031 -6.4489
#>   0.0390  0.5220 -5.8338 -2.3428 -7.0500
#>  -3.4172  7.1635  9.1088 -3.8103 -1.5169
#> 
#> (1,3,.,.) = 
#>  -3.1671 -3.3980  1.5381 -6.9898 -3.7765
#>  -2.8697 -5.6445  1.9030  1.4949  6.0777
#>  -3.7872 -2.1808  9.9312 -6.9776  5.2346
#>  -2.6676  3.9134  2.2968 -0.1042  5.7117
#>  -2.3296  6.9295 -7.1610  3.3373 -2.4823
#> 
#> (1,4,.,.) = 
#>   -0.4921   4.0959  -4.2596   1.5555  -2.7260
#>   -8.1059   3.9995  -4.5888  -4.7081   3.4081
#>   -5.7472   3.9526   1.3324  -0.7352 -10.3185
#>    2.7452  -6.6532  -3.6019  11.2775  -1.8286
#>   -6.9536  -7.7059  -0.7224   8.6867  -1.3742
#> 
#> (1,5,.,.) = 
#>  -1.9383 -1.5019 -0.4628  0.0008  2.6208
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
