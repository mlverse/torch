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
#>    1.2324   8.8174  -1.5462   0.6130   7.7109
#>    1.3946  10.6595   5.2876   5.0241  -2.0523
#>   -6.0834 -10.8384  -9.3663  14.2998   4.5734
#>   -5.3056   3.7470  10.6230   6.9697  -6.0892
#>    1.5859  -0.9842   9.2113   1.7282  -8.0084
#> 
#> (1,2,.,.) = 
#>   -4.0638  -7.5148  12.6518   6.9252   0.0047
#>    4.2911  -5.3973   0.6527   1.4363   2.1434
#>   12.9058   1.7660  -7.4622 -12.4775   0.0706
#>    4.4162   2.0194  -0.4538  -0.0236   1.8438
#>   -2.9550   3.3193  -0.3584   1.8665  -1.2966
#> 
#> (1,3,.,.) = 
#>  -5.9732  4.1739  6.4960  3.2602 -1.1328
#>   7.2139 -0.0260  3.0101 -0.4231  6.2610
#>   2.2509  9.4614 -1.7065 -2.4467 -2.0040
#>   1.7013  4.2153 -1.2033  6.8307 -5.6498
#>   5.3448  6.3970 -2.7104 -1.0270 -0.1497
#> 
#> (1,4,.,.) = 
#>  -3.1492  7.7217 -1.8237 -3.7545 -1.8611
#>  -2.5181  1.8581  6.6200  3.4775 -0.3890
#>  -1.4580 -1.1224  0.8640  7.0670  4.0035
#>   5.9698  1.6036  7.6781  4.3825 -1.9893
#>   4.1709  5.6678 -6.9352 -4.3442  0.2115
#> 
#> (1,5,.,.) = 
#>   4.9063  4.5185 -3.9461  7.0396  4.6552
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
