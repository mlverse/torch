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
#>  4.7697  5.3000 -3.7524 -2.3731 -3.3965
#>   5.5991 -4.4058  0.4269 -0.6533  2.1820
#>  -1.0818  3.6069 -8.2744  1.1278 -3.3102
#>   4.8904 -1.3084 -4.1232 -2.7295 -0.6653
#>   4.1782 -3.9082  0.7554 -0.6538 -0.2483
#> 
#> (1,2,.,.) = 
#>   2.8344  -2.5562  10.8732  -0.3710   0.3178
#>    0.7827   0.8929 -16.7543  -3.1612  -4.6153
#>    0.7418  -0.1575   4.4850  -2.4088   5.2212
#>   -2.3587  -1.8890  -9.4061  -1.4863  -9.1964
#>   -2.5360   1.2115  -7.4703  -1.5376  -0.2323
#> 
#> (1,3,.,.) = 
#> -0.9978 -1.4348 -4.8951 -4.2028 -0.4605
#>   5.2478 -2.1835  4.2022  6.9792  4.5897
#>  -1.1654 -1.5120 -5.3516  3.4597 -4.8292
#>  -7.3127  5.5229 -3.3214 -0.6602  0.1595
#>  -0.1893  5.1490  2.6424 -1.7845  0.1223
#> 
#> (1,4,.,.) = 
#> -5.0081  0.7766 -2.8358 -1.4503 -1.6916
#>   3.6643  2.9153  1.6489  3.1248  7.3186
#>   3.1393  1.4158  2.9056 -9.2365  2.2383
#>   4.7085 -2.0392 -5.7968  6.8822  4.1439
#>  -1.8010  1.5982 -5.5151  1.3478 -1.0607
#> 
#> (1,5,.,.) = 
#>   0.6644   1.5813   5.3769   3.3322   1.1419
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
