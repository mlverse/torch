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
#>   -0.2036  -5.2387   8.0851  -2.6331   5.6137
#>   -1.7405   2.3495  -1.5943   4.0912   6.2619
#>    2.9980  -1.1929   4.8998   0.7546  -4.8484
#>   10.5474  10.3143  -8.8632  -2.9280  -1.5097
#>    2.6008   0.4684   1.1180  -5.8927   2.8399
#> 
#> (1,2,.,.) = 
#>   2.2197  2.7064 -8.0737  7.3916 -0.3449
#>  -7.6722 -9.9320  3.4269 -0.3198 -1.7072
#>   1.8998  0.6788  3.5111  1.1256  4.5414
#>   9.1040 -0.6089 -2.4953  7.2575 -0.4545
#>   7.9574  2.9315  2.2435  5.0770  0.2892
#> 
#> (1,3,.,.) = 
#>    1.0975  -0.5932   3.5058  -0.6274  -0.0140
#>    1.3747  -4.2752  -1.5429 -10.4787   3.4738
#>    4.5281   1.1447  -6.9159  -0.3950   1.3526
#>   -7.7196   1.2532   1.0431  15.2036   3.9769
#>    5.3337   2.6216  -3.5312   0.5003   0.7343
#> 
#> (1,4,.,.) = 
#>   5.4647 -2.9600 -4.1445  0.1424 -2.4491
#>  -4.8389 -2.2437 -9.4212  5.9235 -0.3221
#>  -10.8078  0.1090  1.8906  2.4075  5.5952
#>   3.4861 -6.2856  7.2485 -11.9645  4.4261
#>   1.4189  7.7275 -5.8712 -3.6297 -4.8519
#> 
#> (1,5,.,.) = 
#>   2.5357 -2.9451 -1.3143  1.0238  7.9973
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
