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
#>   2.8108   2.2110   9.4234   3.7619  -2.6977
#>    1.6834 -14.3243   5.7306  -1.3178  -1.0792
#>    7.8661   2.7622  -4.4890   8.5755   0.2790
#>   -0.9980   9.0965   4.9527  -7.8270  -1.1943
#>    0.7582 -10.7094  -5.8700   6.6616  -5.4550
#> 
#> (1,2,.,.) = 
#>   1.2213   3.0518  -5.0605   2.5874   9.0970
#>   -9.8105  -3.2544   4.9163   1.9927   0.8493
#>   -1.7666   1.8339   2.5118  -2.2806   0.1105
#>    3.4305  19.9974  -1.0143  -0.1667   6.2442
#>   -0.2476   3.0913   8.6823  -0.3704  -8.9123
#> 
#> (1,3,.,.) = 
#>  -3.7728  -1.9444   2.3362  -7.3203  -1.8153
#>    0.7267   8.7168   1.8865   5.4098  -1.1499
#>    2.9759   1.2264  -7.2394   0.1390   2.7536
#>    2.4486  -8.7844  -1.7040   1.5610  14.8326
#>    1.5088  -4.8367   6.5121   8.1719  -0.4499
#> 
#> (1,4,.,.) = 
#>  2.5516 -0.2098  1.3444  0.8358 -2.9189
#>  -3.3889 -1.2061 -1.3641  2.6033 -0.3321
#>  -3.1559 -0.8432 -1.4148 -5.1355  0.3059
#>   5.5486  0.1279 -0.0821  5.5491  1.0450
#>   2.1222  8.3609  3.1706  2.2927 -5.7349
#> 
#> (1,5,.,.) = 
#>  -7.1791   2.0701 -10.2450  -5.9132   0.1537
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
