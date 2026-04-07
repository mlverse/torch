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
#>   1.5883   2.0136   3.7273  -0.8186   8.2923
#>    0.0546   1.0296   0.7160   1.0720   6.4666
#>   -2.0295  -0.3412  -5.2736  -3.2269  13.7074
#>   -3.2933  10.4404  -3.3607   1.8553   2.7609
#>   -1.6776   9.9394  -3.1162   3.7445   2.5533
#> 
#> (1,2,.,.) = 
#>  0.9945 -3.9120  2.3245  1.7233 -0.2214
#>  -3.9824 -7.2001  3.3920 -6.1821 -9.6208
#>  -1.7607  6.2147 -5.4509 -1.2277  9.6994
#>  -2.7986  0.8588  8.5668 -2.3888  8.2144
#>  -2.1627  4.8202  6.2715 -1.3767 -1.9053
#> 
#> (1,3,.,.) = 
#>  2.2472 -2.1756 -0.8304 -4.4471  5.8565
#>   1.1556 -2.7510 -4.7975 -0.8474 -0.1950
#>   5.2748  0.5489  3.3284 -7.7335  4.3841
#>  -0.3601  4.0941  1.6218 -1.1863 -0.9655
#>   0.0313  2.9172  4.7762 -7.9718  1.6630
#> 
#> (1,4,.,.) = 
#>  -3.0370  -3.1687  -1.8788  -2.0612   8.4352
#>    2.1638   2.0208   9.9134  -5.3022  -2.3303
#>    0.1838   1.9125   3.3771  -9.8557   2.7169
#>   -0.4408   5.4267  -7.4085  -6.4301  -2.9095
#>   -5.6345   5.8738  12.4981  -3.8691   3.3506
#> 
#> (1,5,.,.) = 
#> -6.6506  3.6499 -1.2937  6.0621  0.3419
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
