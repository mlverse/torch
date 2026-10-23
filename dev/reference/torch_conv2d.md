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
#>  -0.7240  11.0725  -0.6035  -1.4842  -2.1846
#>   -0.8070  -3.2428   1.9998  -0.5917  -4.3230
#>    1.0761 -12.4856   4.3556  -1.8356  -1.1960
#>    3.6953  -0.8164  12.7477   5.6969  -8.3906
#>    0.0432  -3.8889   1.6489  -1.8643  -1.2235
#> 
#> (1,2,.,.) = 
#>   3.2842  -5.8818 -12.1900   3.2968  -2.6274
#>    6.0047 -10.7878  -4.3718 -11.6527  -8.6870
#>   -1.8535  -2.3211   0.7475   3.7820   0.2414
#>   -0.1759   0.5898   0.0282   5.5919  -5.4315
#>   -1.5359   1.6526   0.9338  -0.1814   1.1542
#> 
#> (1,3,.,.) = 
#>   0.0582  11.4998   1.8091   0.6121   0.1396
#>    3.1081 -13.0864   8.9872  -5.2271   2.1040
#>   -1.6240   8.9339  -4.4338 -12.1748  -4.6328
#>   -8.5714  -1.0594   1.6227  -2.8108  -8.8610
#>   -1.1964   5.3756  -5.2689  -6.8722   3.4443
#> 
#> (1,4,.,.) = 
#>   8.9354  -2.4450  -0.7215  -6.1396   1.2061
#>  -13.7589   9.5226 -13.3618   4.9153  -0.9943
#>    4.3959   1.8968  -1.4565  -9.3443   1.7417
#>    2.7245   8.5780  -6.2173  -2.9899   1.1989
#>   -0.1508  -1.8064   2.8828   3.5941   3.3412
#> 
#> (1,5,.,.) = 
#>   4.2588   1.1499   9.1203   5.5934  -0.4149
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
