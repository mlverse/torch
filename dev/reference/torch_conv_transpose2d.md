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
#>  -1.0935  -0.4424   0.5708  -8.3318  -2.0403
#>   -2.7329  -1.4889  -4.7018  -7.1768   2.2714
#>   -1.7138   4.9812   0.4727  -1.3045  -1.2294
#>   -2.9679   4.0413  -0.6651   2.5397 -11.5148
#>   -5.4678   7.1704  -3.7146   3.8988   3.0601
#> 
#> (1,2,.,.) = 
#>   4.4310  -3.9185   3.5189  -4.3852  -4.8247
#>    3.6749  -5.6508  -1.4328  10.4169   2.9991
#>    5.1339  -0.2319  -3.8301   7.1491  -1.1704
#>   -6.1370   3.5996   2.9346  -3.5891   3.1834
#>    6.8208  -3.1793  -0.1841  -2.3128   4.6389
#> 
#> (1,3,.,.) = 
#>  -2.8288   9.0822  -6.4072  -9.6750   3.1297
#>    3.8914  -1.2179  -3.5701   0.4895   6.4029
#>    0.7013  -7.5036   3.7271  -7.6187   0.3383
#>   -8.3227   9.1389   2.4097 -12.4571   6.8097
#>   -3.6668   2.2463  -0.3792   9.2356   4.9102
#> 
#> (1,4,.,.) = 
#> -0.6235  3.8385 -2.6253  1.1176  1.0228
#>   2.5796  0.4527  3.9672  1.8408 -2.3492
#>   1.8406  1.2876 -6.8304 -7.4150 -4.6604
#>  -2.2760  7.4515  9.0459 -0.0817  8.4316
#>  -4.8317  0.0773  9.4504 -0.3671 -1.2995
#> 
#> (1,5,.,.) = 
#> -2.8875 -5.0960  8.0172  5.3620 -0.6937
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
