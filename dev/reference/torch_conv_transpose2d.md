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
#>   0.1607  4.6910 -7.4690 -6.7417 -5.5273
#>  -4.5687 -10.7836  4.7833  7.2426  2.6057
#>  -4.4759  4.3671 -4.0779  3.1503 -1.1983
#>   0.4492 -3.3068 -8.9531 -10.3334  1.5610
#>  -4.6656 -5.8246 -2.2416  5.6884  3.6860
#> 
#> (1,2,.,.) = 
#>  -8.6384 -5.0889  3.3040  3.4550 -0.7699
#>   5.1274  1.0170 -1.4510 -9.4795  1.5801
#>  -4.0065 -2.2772 -0.2420 -3.8777  7.3903
#>  -7.2377  7.6565 -4.3706  5.3025 -3.4918
#>   3.1919 -0.6855  7.0842 -4.8546 -5.9814
#> 
#> (1,3,.,.) = 
#>   0.1999  4.4698  3.3170 -5.7602 -2.5175
#>  -10.0386 -11.6342 -9.6245 -0.4669 -1.6437
#>  -8.1160 -10.6606  2.6087 -1.2674  2.1921
#>   3.1946 -3.1315 -2.5165 -1.4045 -5.1772
#>  -8.8301 -9.6933 -11.7355 -2.8998  2.4321
#> 
#> (1,4,.,.) = 
#>  -11.8617  -2.7950   2.7954   0.8107   0.4487
#>    4.9980 -10.2744   1.7282   1.0110  -0.1207
#>   -0.3947   6.4644  -4.3089  -8.9102   3.3100
#>   -4.3263  -6.6685  -9.0787   3.4559  -1.8163
#>   -2.2087   2.7752   1.0733   0.1511   1.5020
#> 
#> (1,5,.,.) = 
#>  -3.7247  0.1315  4.6377 -0.6707 -1.6237
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
