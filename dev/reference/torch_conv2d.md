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
#>  13.3449   2.2602   3.3576   2.9957  -3.7987
#>    0.1704 -10.6018   7.7173  -0.0100   1.6155
#>   -2.5196   9.3817  -0.4614  -1.2028  -1.8416
#>    1.9837   4.9448  -2.1587  13.2860   3.2323
#>    3.7709   2.1049  -0.4726  -2.0268   1.6148
#> 
#> (1,2,.,.) = 
#>   3.8136  -1.5317   4.3833  -2.1729   1.6123
#>   -7.0623  -6.6513  -2.4790  -2.8623  11.2867
#>   -0.9251  -1.5288 -10.2138  -2.4739   3.8732
#>    3.1200   1.0103  -4.9284  11.1622   2.1669
#>    5.0849  -3.0324 -14.2382  -0.1897   3.1805
#> 
#> (1,3,.,.) = 
#>   2.7556  -8.6872  -1.6384  -0.8633  -4.1284
#>   -2.5638   3.9455  -0.9990  -6.7691  -3.8235
#>   -3.3906   9.4090 -11.7176   4.3319   0.1130
#>   -4.2449 -10.4946  -4.0518   2.9207  -4.6105
#>   -3.7123  -0.5070   0.7194  -2.0660   5.1069
#> 
#> (1,4,.,.) = 
#> -3.2952  1.2427 -2.8074  1.4936  0.9453
#>  -0.1221  7.9141  2.6897  1.7724 -0.2595
#>  -3.8556 -8.8503 -0.9464 -0.6839  7.1642
#>   3.5045  4.6809  2.9695 -4.5160 -4.7103
#>  -0.9813 -2.6962 -4.1669 -7.7335 -1.3427
#> 
#> (1,5,.,.) = 
#>  0.5580  1.8198  2.5776 -4.7725  0.4379
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
