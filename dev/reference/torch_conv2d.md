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
#>   -0.8484  11.7794  -4.7053   0.8430   2.3074
#>   -1.1317   2.4561  -6.8602   7.9688   3.3617
#>    3.1368   3.3140  -8.8225   8.0208   0.0826
#>   11.3924   5.6579  -0.6069  -2.2142   0.4060
#>   -3.0696  -6.5021   6.1034  11.2740  -6.1046
#> 
#> (1,2,.,.) = 
#>    3.8836   2.1680  -9.6702   5.9918   4.9346
#>   -0.3018  -7.1704  -5.7350  -2.0562   1.6071
#>    3.2166  11.6757  -9.8445   7.2739   3.8762
#>   -1.0763  -3.0662   0.3300   1.9224   0.2671
#>   -5.6926  -6.1345  -0.9312   6.1341   3.9757
#> 
#> (1,3,.,.) = 
#>   -7.1163   4.6281   7.4040  -3.7725  -4.6868
#>   -0.5755   0.1386  11.3265   1.3533   3.0436
#>   -2.0801   3.8134  -4.0164  -3.6908  -3.3292
#>   -0.5674   8.0078   0.7564  -3.9957   3.9505
#>    2.7939   1.3009  -0.2714   3.8267  -5.0474
#> 
#> (1,4,.,.) = 
#>   -3.5298  -3.3535  -6.4959  -5.2972  -2.6767
#>    1.2225   4.4047   9.6100   4.4116   2.2745
#>  -13.3603   1.6013  10.3959   2.6876  -1.4407
#>    2.8921   0.2914  -0.5578   3.1921   5.8086
#>    5.2430   4.0024  -0.1065  -5.0596   1.2529
#> 
#> (1,5,.,.) = 
#>    7.1494  -5.1789   8.2081   1.8233  -0.5589
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
