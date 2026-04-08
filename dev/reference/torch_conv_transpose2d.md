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
#>   0.2399   0.5421   1.8677  10.7031  -6.3145
#>   -3.0306   1.2152  10.4579  -1.0499   1.2417
#>   -4.5494   1.5495  -2.1143  -3.2378  -2.4280
#>   -3.0823   7.9046   2.2419  -4.7394  -0.8995
#>    1.6387 -11.3128   0.0510   6.1104  -3.8771
#> 
#> (1,2,.,.) = 
#>  -0.2691   3.5763   1.6145  -4.5858   2.5135
#>   -4.1448   6.4847   9.6776   0.7785   4.3830
#>   -0.2828 -10.0884   4.5824   0.1583   0.1644
#>    4.2012   0.3660   8.2016  -3.6213  -3.2733
#>    2.2726  -8.7337  -5.3229   6.2347   4.4880
#> 
#> (1,3,.,.) = 
#>   4.1465   7.9850   0.4438   1.3689  -4.2454
#>    0.9764   0.3557  -4.7919   1.6796  -8.2161
#>    8.5645   3.4660   7.0431  -2.1791  -0.1184
#>    2.6463  -2.8778 -11.3364   0.8277  -2.3098
#>    3.9177   2.3592   0.5266   4.0583   2.9988
#> 
#> (1,4,.,.) = 
#>   4.4381   9.3154   6.8195   1.2961  -3.3365
#>   -1.6137  -0.6024   3.5872   1.1148   2.3965
#>    4.9909   2.9668   5.1601  -0.0913  -1.0335
#>    3.4894   9.0293  -4.4079 -10.3999   5.6621
#>    3.9255 -10.1176   2.4243  -7.1757  -5.9789
#> 
#> (1,5,.,.) = 
#>  7.4429  4.7305 -1.6415 -2.3722 -0.5892
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
