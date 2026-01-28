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
#>  -0.8407  3.7290  3.1363 -4.7879  1.2158
#>  -1.1871  9.4472 -6.0914 -4.4516  5.9892
#>   0.4040 -3.1259 -2.0503  4.3236 -8.2150
#>   1.4634 -8.4729  2.9854  5.9527  2.6010
#>   1.5735 -2.4310 -1.0331 -1.6351  0.8489
#> 
#> (1,2,.,.) = 
#>   -2.9002  10.1765  -5.1925   4.8948  -6.2587
#>   -5.5811   2.2495   2.6905  -6.3342   4.1927
#>  -17.7492   7.0974   4.4134   0.2659  -1.1007
#>   -1.4931   9.0821   3.0450  -5.9763   6.7160
#>    2.1640  -8.2322   4.9624   1.1389   0.7195
#> 
#> (1,3,.,.) = 
#>   -3.4626  -5.7427  -0.5143  -1.4131   3.5591
#>   -1.8583   4.8855   0.4058   5.8278 -10.6326
#>   -3.9651   5.3754  -5.2264  12.8814   4.6063
#>   -1.5647  -6.3877  -6.5254  -4.0190   2.6575
#>    2.9068  -1.0847   2.0366   0.7250  -1.9032
#> 
#> (1,4,.,.) = 
#>   -5.3709  14.7637  -2.7547   7.8934   3.6280
#>    5.6161   3.2363   0.8443  -0.6640   2.9025
#>    4.5445   2.8210  -4.1953  10.1717  -2.1754
#>   -7.3542   7.0889  -5.6582   0.1439   3.0632
#>   -0.9593   1.1425  -2.6357   1.4215  -3.1388
#> 
#> (1,5,.,.) = 
#>    6.7251   4.2107   1.9729  -1.9624  -1.7752
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
