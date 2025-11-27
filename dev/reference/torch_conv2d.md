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
#>   0.4641  1.3740 -3.9259 -3.9946 -3.4468
#>   0.1268 -4.2987 -6.9801 -3.1654  0.6084
#>  -1.8821  3.0813 -3.3739 -7.9259 -0.3882
#>   4.0487 -6.1655 -3.3938 -2.9737 -5.9149
#>  -1.4092 -0.6342  1.9087  3.2619  3.0444
#> 
#> (1,2,.,.) = 
#>  -0.9269 -1.0605 -3.4698 -6.0376  1.8129
#>   2.2575 -18.5043 -0.3769 -11.8291  1.5342
#>  -6.2299 -8.1822 -4.5282 -1.3443  3.2156
#>   0.6714 -2.8659 -0.7305  3.5440  2.2857
#>   2.1913  4.7044  1.7431  2.4486 -0.9694
#> 
#> (1,3,.,.) = 
#>   -7.8156  -1.6486  -3.9072  -1.4039  -0.1077
#>    4.7875   1.3942  -4.7671   1.5254  12.9735
#>   -5.3743  -7.6872   3.0487  -7.4420   5.7060
#>    6.3413  -1.4198   2.0300  -4.9250  -3.7829
#>    8.8739   4.6501  -0.2639  -0.7494   5.0350
#> 
#> (1,4,.,.) = 
#>   2.3175  1.7178  1.6540 -1.2577 -4.1284
#>   1.1674 -6.9365 -3.5118  3.4302 -1.2432
#>  -3.7021  0.0611 -1.7123 -2.1319  3.7021
#>   0.1009  4.5113  0.0058 -1.3735  3.3064
#>   6.7547  5.8293 -4.4210  3.0263  0.5070
#> 
#> (1,5,.,.) = 
#>    0.6239  -0.7415  -5.6398  13.5732  -3.8377
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
