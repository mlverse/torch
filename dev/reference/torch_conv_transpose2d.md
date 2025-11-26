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
#>   1.8303 -11.2874  3.3733  0.2709  0.9880
#>   7.6803  2.2565 -0.2246 -8.2342 -6.7381
#>  -1.0735  6.1311 -2.9594 -2.6886 -0.1689
#>   4.9099  1.2992 -5.4287 -4.9426 -0.7018
#>   1.5231  4.9825  1.0313 -4.4070  3.3866
#> 
#> (1,2,.,.) = 
#>   -1.9593  -2.1154   2.5091  -3.4639   1.8554
#>  -10.2957  -0.2928  -1.8167   7.3336  -0.9892
#>  -10.4044  10.8187  -4.5702  -7.4417   5.3074
#>    1.1945  -3.7934   5.9631   9.7878  -8.1612
#>    4.4473  -2.5373  -0.2191  -2.2456   5.0231
#> 
#> (1,3,.,.) = 
#>    1.9918  -9.6716  14.4059  -7.0772   5.3224
#>   -0.6213  -4.5868  -2.6232   9.7715   0.4291
#>   -2.0728   4.9389  -0.6740   8.2820  -9.4965
#>   -6.3088   9.6000  -2.0550   0.8729   0.6695
#>   -1.6789   5.9015  -5.8384  -3.8800   3.9748
#> 
#> (1,4,.,.) = 
#>  -7.6244  5.1717 -3.5692  2.5199  0.4549
#>  -3.5414 -5.3360  3.6624 -0.8218  6.8044
#>   4.8826  7.3071 -2.6404 -2.3285  4.7639
#>   0.1627 -1.3179  2.6029  0.0722  2.9666
#>   2.0288 -12.0777 -3.7577  7.9287 -1.6862
#> 
#> (1,5,.,.) = 
#>    5.9670 -15.4129   5.5230  -3.9877   9.1783
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
