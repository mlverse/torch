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
#>  -7.7494   8.1423  -6.1057 -10.2915  -5.7179
#>   -7.0567   3.4465  -6.0116  -0.3519   4.0292
#>   -0.5551  -0.0635  -7.2822  -1.4338  -3.2077
#>    1.9234  -1.0861  -7.7315   0.2196  -5.0887
#>    5.4355   2.1023   1.7313   1.3198   1.8362
#> 
#> (1,2,.,.) = 
#>   0.8522  -4.9653   4.2714   0.2079   1.5278
#>    2.7732  -7.6999  10.5087  -4.3651  -4.9035
#>   -4.3510  -7.3528   0.1685   3.3446  -1.9979
#>   -2.6199 -10.2127  -6.7161   4.9175  13.4292
#>   -2.0867   2.8047  -8.3253 -13.4221  -0.1731
#> 
#> (1,3,.,.) = 
#>  -4.1580  -2.6997 -10.1411  -0.9900   0.5180
#>   -6.4686   1.8151 -10.7226  -5.1307   5.0393
#>   -5.0290   8.4345   3.9712  11.1979  -6.6263
#>    2.0162   6.2667  -2.0743   8.0876  -4.0932
#>    4.3982  -2.5035  -0.5589   6.1900   6.9428
#> 
#> (1,4,.,.) = 
#>  -0.2947  -5.8524   4.0977   7.2346   6.2801
#>    5.2394  -3.8174   0.4517   1.8745   5.8864
#>    1.1982  -4.5570   0.2187  -1.5792   0.7223
#>   -1.6075  14.0285  10.8683   3.1419 -11.5898
#>    1.4109  -1.7402   0.8191   8.9446   3.2676
#> 
#> (1,5,.,.) = 
#>  -6.0492   5.7844  -0.2270   3.0227   2.9007
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
