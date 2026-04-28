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
#>   2.5568  -1.8705   2.5651  -2.8518  -0.4414
#>   -6.4169   7.6458   4.3671  -4.9875  -3.2369
#>   -3.1286   2.6792 -12.7062  -5.5724  -8.7022
#>    2.2325   1.8812  -3.8924  -0.1773  -2.4867
#>    3.2300  -3.1472  -5.2938  -5.6013   1.7491
#> 
#> (1,2,.,.) = 
#>   2.8212  -1.8808  -4.3737  -1.8916  -3.5754
#>   -1.9973   5.3080  -5.5791  -9.7201  -8.9990
#>   -2.0654  -6.1619  -0.6746 -11.2846  -9.5520
#>    1.7883  -6.1783  -2.7848   0.9098  -9.0128
#>   -1.1351  -3.0385  -6.0454   1.0130   5.4653
#> 
#> (1,3,.,.) = 
#>   3.4252  -4.1577  -1.6184  11.1418  -1.3712
#>    5.3313  -8.7328   2.1915   9.9304   4.7661
#>    7.5666  15.2094  -8.6994   3.0033   3.3304
#>   -7.6365  -7.9769  -0.5043   7.9092   4.4929
#>   -4.0548   3.7929  12.1501   2.7287  -2.3606
#> 
#> (1,4,.,.) = 
#>  -2.7799  -2.8323  14.1922  -6.5315   4.9636
#>    4.4233 -19.4052  -7.2880  -5.9969   7.4471
#>    5.2161  -7.7022   5.8766  -9.0684   7.8324
#>    5.4072  -3.8466  -7.1176  -6.3199   1.9464
#>  -10.2469   4.5499   4.1510  10.8789  -3.5523
#> 
#> (1,5,.,.) = 
#>   3.2937  -1.8881  11.0727   1.5660   1.9271
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
