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
#>   2.5909 -3.7137 -6.8142  5.5038 -2.7152
#>   7.2341 -3.3789 -6.9240  3.7613 -7.3706
#>   6.1062  2.1368  1.7795  0.1169 -1.3454
#>   4.5130 -7.9261 -1.0433 -5.3525 -3.4081
#>   0.9070 -2.5694  0.1647 -1.4843  1.3168
#> 
#> (1,2,.,.) = 
#>   6.4365  1.4395 -0.6299 -0.2499 -0.8910
#>  -1.1304  8.1613 -2.3421 -5.3245 -7.4609
#>  -2.3274  3.3081  6.2062 -4.6634  1.8090
#>   3.8308  2.1909 -3.5342  2.7964 -1.0309
#>   4.6139 -1.4198 -4.9413  3.8763  4.4351
#> 
#> (1,3,.,.) = 
#>   -1.2782   5.9888   0.5559   6.1867  -1.8792
#>    0.5653   5.7922   0.9955   5.3422   1.0513
#>   11.1490  -0.4075  -7.0774   7.0208  -3.7294
#>   -1.0032  -3.4300  -0.1441   4.2743   1.7147
#>   -0.5380  -0.7169   0.0637  -4.6991   3.4474
#> 
#> (1,4,.,.) = 
#>  -0.7880 -1.5967 -0.3696  1.8832 -3.0850
#>   5.7493 -5.7501  3.2027 -8.3215 -5.2496
#>  -1.2366 -2.9153 -0.6847  2.6704 -3.4354
#>  -7.0824 -3.3216 -4.6578 -12.4554  1.8372
#>  -5.8873 -1.8201  5.7512  1.0787 -2.1351
#> 
#> (1,5,.,.) = 
#>   6.1749 -8.4013  4.7412 -1.7956  2.3089
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
