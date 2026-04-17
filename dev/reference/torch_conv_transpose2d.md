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
#> -3.3822 -3.3515 -4.2513 -1.6636  1.8829
#>   2.9857 -1.4560  3.7499  1.9207  2.2275
#>  -4.0400 -0.4125 -0.3261 -4.5278 -1.2297
#>  -6.3825  0.3016  2.5969  1.9160  4.3466
#>  -6.1642 -3.5140  8.2325 -0.1915  6.4560
#> 
#> (1,2,.,.) = 
#> -4.5782 -0.5896 -0.3965  8.1687 -3.1086
#>  -0.8970 -4.5405 -1.6437  2.1635  3.1329
#>  -4.8291 -3.2869 -3.0140  2.4493 -2.1883
#>  -9.4911 -8.5263 -4.1835  8.2188 -4.3345
#>  -0.0697 -0.3139 -0.4256  7.9921  3.5552
#> 
#> (1,3,.,.) = 
#>  -4.9178   0.2995   2.3084   3.3539  -2.1736
#>   -2.5597  -4.2289   8.6533  -1.1740  -1.3730
#>   -2.3660  -4.1405   4.1003   4.6370   0.3950
#>    1.9975 -10.3441   4.5621   6.8783  -1.6924
#>    3.2861  -5.8540  -0.2902   8.4014  -4.3992
#> 
#> (1,4,.,.) = 
#>   3.2084  -0.7449   1.1011  -0.1690  -1.6332
#>    1.2940   3.5491  10.2530  -2.2509  -0.2032
#>   -1.4902  -6.2137   6.6378   2.4094   0.8047
#>    0.4424  -0.3985   6.5861  -2.3878   3.7930
#>    6.4759   0.4138   7.2219  -2.5361  -9.3960
#> 
#> (1,5,.,.) = 
#>  2.2023 -0.8663  0.8391  0.4768 -1.0107
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
