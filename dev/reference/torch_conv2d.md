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
#>   2.4678  -3.0818  -6.0432  -6.3858  -2.0142
#>   -1.0836   6.5509  -2.2536   3.6771   3.4273
#>    0.8218   6.0958 -10.1673   7.2820  -1.4127
#>   -1.6564   1.4003  -1.5757   5.0860   0.9680
#>   -2.7120   9.1229   2.5484   1.2684  -5.8214
#> 
#> (1,2,.,.) = 
#>  1.3388  4.0874  3.2577  1.5843 -1.8104
#>   2.7317 -2.4100 -5.1530 -4.4735  2.5591
#>   4.0375  0.0324  2.6693  8.0237  5.1939
#>  -3.7119 -5.3282  1.7129 -3.5503 -4.8666
#>   4.1477 -1.6326  6.4378  0.7396  0.6745
#> 
#> (1,3,.,.) = 
#>  -1.1084   2.2686   0.1174   0.0455  -3.5622
#>    2.2213 -12.3502   0.4885   6.1404 -10.7105
#>    1.5391  -8.1451  13.4235   0.5220   2.0442
#>    6.6972   0.2183  -6.9360  -4.8862  -2.6123
#>    7.5835  -9.5066  -2.7196  -4.6971   3.5250
#> 
#> (1,4,.,.) = 
#>  -2.9494   1.1605   1.8065  -0.6736   2.1968
#>    2.0713  -2.0703  -1.6653   1.6167 -12.7556
#>   -1.0909 -14.1376  -2.3783   5.1737  -4.5994
#>   -3.1688 -10.3377  -2.3665   2.5565  -1.6477
#>   -1.2621   6.7879  -9.9960   1.1236   8.2855
#> 
#> (1,5,.,.) = 
#>   0.7717   6.9559   4.8764   9.6109  -2.6671
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
