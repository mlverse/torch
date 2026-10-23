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
#>   2.3679   3.8127  -8.0567   1.6051   2.8208
#>   12.2025  -9.6387  -2.3286   2.4999  11.7723
#>   -3.8258  12.8081 -14.8360   0.2033  -3.1897
#>   14.1829 -12.2520  10.6937   1.0531 -11.1117
#>   -1.7020  10.4313  -2.9523  15.2137  -4.2985
#> 
#> (1,2,.,.) = 
#>   0.6122   2.6235  -7.0876  -9.3953  -5.4168
#>    4.5650  18.8916   6.6637   1.6567  -3.6095
#>    0.8082   1.8213   3.8717  12.8500   7.0082
#>    0.0401  -1.3546  -6.0748  -7.7480  -3.6321
#>    0.2826  -4.8308   7.0654   2.2459  -0.6822
#> 
#> (1,3,.,.) = 
#>   1.7919   5.8726   6.5746  -7.6637   5.1110
#>    5.6794  -3.9490   0.2715   9.7447   2.9631
#>   -7.9430   9.2149 -17.4690  -7.4045   1.8279
#>    7.7967 -12.0958   5.4535  -3.3965  -8.2784
#>   -3.4542  10.8380 -22.5166   9.3408  -0.3659
#> 
#> (1,4,.,.) = 
#>  -1.6371 -10.9196   1.5852  13.2970  -2.8395
#>   -4.4670 -10.0543   2.0571   7.8619  -5.3115
#>   -0.4178  -9.5071   3.6985  -6.7333  -4.2251
#>   -2.9714  -7.7633  -8.1378  -0.9960  -7.7290
#>    6.4559  -6.4197   1.5058   2.4426  -0.1821
#> 
#> (1,5,.,.) = 
#>  -0.6097   9.1704   5.0820   0.7676   2.4554
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
