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
#>   4.2720   1.2930  -8.9640  -1.9322   0.2944
#>    1.7184  -1.1268   8.9841  -0.0601  -1.8879
#>   -0.6330  -4.0405   1.9113   4.7155  -0.6735
#>   12.0107   5.2963   7.2530   3.0710  -5.0957
#>   -0.7873  10.6636  -3.2843   9.1634   1.3898
#> 
#> (1,2,.,.) = 
#>  -3.2247   5.7247   3.6323  -4.1602  -2.6252
#>   -2.0790   0.1954 -14.4073  -0.9616  -0.2687
#>   -1.7054   3.0026  -1.2097  -6.2157  -2.3690
#>   -9.5177   0.4839   3.2163  -0.0856   7.0503
#>    0.2580  -5.1183  -6.4691   5.4392  -3.8402
#> 
#> (1,3,.,.) = 
#>  1.4925 -3.3908 -0.3899 -7.9646 -5.7269
#>   2.5538 -0.8029  7.9072 -3.0752 -8.6602
#>  -1.0829 -3.4249  1.0587  9.1821  6.5891
#>   0.3525 -4.9481 -6.6116 -0.0338 -4.7736
#>  -1.5900  0.7541 -3.6835  0.5872 -3.4031
#> 
#> (1,4,.,.) = 
#>   2.8854   0.3935  -6.8534  -4.9935  -5.9864
#>    3.0937  -8.7619  -4.6101  11.7817  10.2479
#>    2.6317 -10.0136  -1.9180  -0.8477   0.2072
#>    3.6721  -0.2251  -0.9240   1.1866  -6.0050
#>    2.0369   1.4092  -8.8751   2.6707  -3.1422
#> 
#> (1,5,.,.) = 
#>  -0.1801  -3.6189  -2.2759  -4.9606  -1.4031
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
