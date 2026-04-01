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
#>  -0.9114  4.0067 -0.1058  7.1184  2.3285
#>  -0.4071  1.6558  4.9427 -1.2101 -1.1527
#>   6.1771  4.6281  0.9810 -2.0806 -7.3214
#>  -5.2958 -0.9375 -2.3731 -0.4616 -1.9583
#>  -1.6314 -1.9926 -11.0085  1.8378  4.6172
#> 
#> (1,2,.,.) = 
#>  -5.2164  0.2264 -3.1485 -0.3796  0.9353
#>   2.1610 -2.5620  0.2673 -2.8989 -4.2255
#>   2.1976  2.6859  2.1438  1.3821 -7.6379
#>  -6.3367 -8.0948 -2.6256  7.7603  6.8547
#>   2.0948 -0.8638 -8.6193 -5.6136 -2.1242
#> 
#> (1,3,.,.) = 
#>   -1.1263   6.3474  -0.8600   3.5267   3.8522
#>   -7.4842  13.4314  -3.2505   0.7930  -2.0994
#>   -5.7884   0.0502   6.2254   1.7377  -3.9219
#>   -3.8929  -3.1920   1.0391  -4.4651   5.1331
#>    0.4861   4.2428  -0.2800  -0.5040   2.0641
#> 
#> (1,4,.,.) = 
#>    4.7249  -1.5683  11.2133   0.7587   1.7740
#>   16.3395 -17.5542   2.5269  10.6019   3.8298
#>    1.7746  -2.3614  -5.7709   3.0925   1.1119
#>   -7.1071  12.0314  -2.6494  11.6568 -12.5937
#>   -5.7488  -0.6086   2.9445   3.4629  -8.5634
#> 
#> (1,5,.,.) = 
#>    1.7089  11.5619  -8.7493   2.5979   0.9968
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
