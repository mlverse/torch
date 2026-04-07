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
#> -4.2370 -3.6127  5.1472  1.2118 -1.5317
#>   0.5986 -3.0771  4.8290  3.4899  0.3958
#>   0.4797  4.4079  3.8144 -1.6469 -3.6098
#>   2.7908  3.6602 -0.7212 -2.9961 -1.0662
#>   1.5045  3.3499  0.7596 -5.7486  3.2849
#> 
#> (1,2,.,.) = 
#>  -4.4584  -2.9377   4.3145  -2.2235   0.0805
#>   -6.9887   6.5734   1.1549   0.8066  -3.3197
#>    3.7536  -2.5909  -4.4329  -0.1316   4.5963
#>   -1.6968   7.5155   2.3971  -9.1584  12.0097
#>    0.3292  -6.5200   8.8614   1.0059   1.1456
#> 
#> (1,3,.,.) = 
#> -12.1778   4.0776   1.2340   7.3931   7.1915
#>   -7.4346  10.5340   2.0196   1.1674  -2.3479
#>   -1.4328  -2.5073  -9.6106   7.7865   0.0637
#>    2.4655  -6.4550  -9.2970  16.4908  -4.2335
#>   -3.0526   4.1907   3.8950  -5.1883  -6.3607
#> 
#> (1,4,.,.) = 
#>   5.4994  -0.1357   6.4793   7.3422  -1.1044
#>   -6.3665  -1.4039   1.7466  -0.8308  -7.4383
#>    0.8762   8.2613   1.7204  10.4363   6.3032
#>    7.7669   5.0092  -7.5004  -7.8433   2.6389
#>   -1.3536  -6.8323  -2.1866  -7.9315   2.6752
#> 
#> (1,5,.,.) = 
#>  -7.2375  -0.0492  -0.4529   6.0755   2.3091
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
