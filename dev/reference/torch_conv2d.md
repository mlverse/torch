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
#>   3.3482   9.9584   0.8151   0.3613   1.5733
#>    2.2991  -4.8980   4.0645   3.9590   2.5677
#>   -3.5684  -5.3012 -10.8799   5.2710  -1.8406
#>    2.0839   4.8134  -8.4365   3.4258  -1.4892
#>   -1.4328  -2.4763  -1.0854   1.1329   1.9401
#> 
#> (1,2,.,.) = 
#>   2.6239   5.8813  -3.4135   9.3408  -0.5784
#>   -6.3351  -0.0882  -5.9926   0.5678   2.0643
#>   -9.0330   6.4924 -17.3853   0.1769  -7.3313
#>    2.9491   1.3456   7.7441   3.0640  -1.2176
#>    0.1415   4.8911   4.8228   7.1259   1.6812
#> 
#> (1,3,.,.) = 
#>   4.3772   8.1339   5.7943   5.2957  -0.3651
#>    1.9790 -19.3569   9.4953  -8.9212   3.0552
#>   -0.9718   3.5174 -10.0545  -0.6636  -7.2883
#>    1.8561  -3.4602  15.2542   5.3643  -0.9712
#>   -1.3104   6.0169  -4.8789  -2.4676   2.2442
#> 
#> (1,4,.,.) = 
#>  -3.6866  -8.0544  -5.5202  -1.6524   1.3324
#>    5.9863   8.5123  14.3045 -10.9622   6.0383
#>   -4.9073 -11.9540  -1.1541  -7.1887   3.7354
#>   10.1205   0.9657   7.9430 -14.9926  10.3861
#>    0.6972   5.5834  -3.9504  -1.9947 -10.3595
#> 
#> (1,5,.,.) = 
#>   6.5994   2.0329  -3.0536  -4.2196  -5.1019
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
