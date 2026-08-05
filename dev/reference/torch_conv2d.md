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
#>  -0.9428   3.8259   1.4368  -0.9112   1.3183
#>   -6.5740  11.6307   3.0352   2.7762  -3.7181
#>   -1.1260  -4.7792   1.6751  -4.4717   3.7162
#>   -9.0454  -0.4499   4.7990  -2.2853   7.5071
#>   -9.2298   3.0016  -2.5040   3.2285  -6.3920
#> 
#> (1,2,.,.) = 
#>  8.7345 -5.5936  0.8074  1.0282 -5.3895
#>  -1.0945 -0.9896 -1.0029  5.8601  3.0622
#>  -2.6546  6.3720  0.5335  7.7028 -5.2363
#>   2.0313  3.9019  0.5332 -1.9068  3.8817
#>   9.8363  4.0081  9.7423  0.8780 -4.9758
#> 
#> (1,3,.,.) = 
#>  -1.9448   4.5675   0.4353   3.0257  -1.6347
#>   -4.1658   6.6160   0.5007 -10.5838  -5.2702
#>   -7.1434   3.3123  -2.3852   2.2018   1.6949
#>    7.9895  -0.4637   3.7525  -3.0932  -4.9549
#>    2.3476   0.8752   0.0714   0.4675  -0.1085
#> 
#> (1,4,.,.) = 
#>  -7.6320   0.7744   0.7220   2.6841   1.5338
#>  -11.0091   9.1374  -9.7719   0.1780  -2.7040
#>  -11.9733   1.5472  -3.5484  -0.7390  -3.4213
#>    3.0993  -0.5783  -2.3022   7.7085   4.0699
#>   -3.9862   2.3800   2.0394  -0.7012  -4.6142
#> 
#> (1,5,.,.) = 
#> -7.6036  1.4394  0.5927  2.3163  5.5025
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
