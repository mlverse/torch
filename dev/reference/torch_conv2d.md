# Conv2d

Conv2d

## Usage

``` r
torch_conv2d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias tensor of shape \\(\mbox{out\\channels})\\. Default:
  `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a tuple `(padH, padW)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv2d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

See
[`nn_conv2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
filters = torch_randn(c(8,4,3,3))
inputs = torch_randn(c(1,4,5,5))
nnf_conv2d(inputs, filters, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#>   2.4619   1.0427   3.9100   9.1037   1.7858
#>    2.1477   3.5842  -2.4950  -5.7085   3.6408
#>   -3.3713  -0.3423  -5.8061 -10.8161  -5.5744
#>   -1.3010   1.1257   0.5983  -5.7756  -8.1040
#>    6.0246  -1.9215   1.6777   2.7939   6.0435
#> 
#> (1,2,.,.) = 
#>  4.2744  6.4611  1.8974 -5.7546 -4.7490
#>  -0.6805 -5.2045  3.6278  3.9185 -0.2216
#>  -6.4347  3.6843 -1.2099  2.6912 -2.9149
#>  -0.2993  1.0291 -5.4167  2.8033  9.2936
#>  -4.9939  0.6138  2.9742 -2.5785 -1.1793
#> 
#> (1,3,.,.) = 
#> -2.2005  4.8523  9.6843 -6.4589 -3.4289
#>  -3.4141 -1.5731 -4.1544  4.7103 -4.7386
#>  -6.5193  1.1523  0.2136 -2.8148 -1.3168
#>   0.8481  1.7306  2.0076 -0.5568  0.1187
#>   3.3355 -3.9707 -2.8957  5.2809 -1.3610
#> 
#> (1,4,.,.) = 
#>  -1.3093  -0.6343   7.2440  -1.8230  -2.8659
#>   -4.6020  -5.7392 -12.6211  -4.1286  -1.5684
#>   -4.4040  -1.3489   1.6791   5.7545   5.1946
#>    7.8023   1.0847  -0.4797   2.3030   3.6312
#>    4.0206   1.3185   2.1458   6.2482   6.2649
#> 
#> (1,5,.,.) = 
#>  -4.6755   0.3798  -3.4565  -5.1282   6.5384
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
