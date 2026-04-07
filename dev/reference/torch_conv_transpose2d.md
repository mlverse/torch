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
#> -2.0615 -1.7128  0.2308  4.0375 -2.9413
#>  -3.0533  1.1705 -5.2886  0.7857  1.1230
#>  -7.7853 -3.0942  9.8391  5.0720 -9.6698
#>   2.6185 -0.0275  0.0315  7.9517  5.8073
#>  -0.3540 -0.9238  3.4770 -3.5239 -0.3537
#> 
#> (1,2,.,.) = 
#>   0.9712  -4.1934   1.4831  -0.5591  -2.3312
#>   -0.6513  -2.6040   1.2755   3.3644   0.3838
#>   -2.1787  -0.0034  -6.8394   3.6585   7.9198
#>   -3.9931   0.5898   3.5642  -3.0192 -14.0525
#>   -2.4967  -2.0182   3.8355   1.8014  -3.3035
#> 
#> (1,3,.,.) = 
#>   2.2548   1.9736   4.6715  -3.2449  -1.1279
#>    1.8462  -7.3408  -6.3616  12.4998  -2.4183
#>    6.2202  -4.4156 -13.8572  -3.4962   6.4126
#>    0.7666   6.7473   2.7791  -3.5506  -0.7965
#>   -2.7801  -3.2673  -0.8053  -2.6395  -0.2864
#> 
#> (1,4,.,.) = 
#>   0.0894   3.4494  -5.0082  -2.9312  -0.8002
#>   -6.8125  -3.8944  -0.1613   4.9080   2.3098
#>   -1.3256   1.8383  -4.2354  10.6527   3.4793
#>    3.1593  -2.6363   4.6690  -1.8700  -2.8478
#>   -3.1674  -0.0328   3.7406   1.9845  -2.9171
#> 
#> (1,5,.,.) = 
#>   0.2892  -1.1432  -3.7838  -2.4857   1.1523
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
