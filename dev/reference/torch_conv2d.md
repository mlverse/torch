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
#>  -0.1916   4.1709   7.7934   4.8607  -2.7004
#>   -4.7872   1.1107   5.0125   9.0998   1.7769
#>  -13.6676  -1.9811   3.8752  -4.9258   0.8190
#>    0.1432   0.6427  -1.5304  -0.7529   0.9800
#>   -4.3836  -0.7106   2.7979  -1.8438   0.6805
#> 
#> (1,2,.,.) = 
#>  1.3783  7.0314 -0.6006  0.5864  1.5785
#>   0.9862 -7.2717 -5.0353  2.6253  5.9059
#>  -1.4720 -2.8394 -2.1506  1.0079  1.5948
#>   4.8332 -4.0200  2.9447  0.1547  2.5665
#>  -7.5093  5.7303  4.5750  1.5434  0.1963
#> 
#> (1,3,.,.) = 
#> -2.5582  5.0959  3.3088  7.7429  1.3810
#>  -9.7839 -6.1309 -9.7562 -4.0834 -4.9550
#>   7.7681  3.6095  5.5268 -0.7368  1.7698
#>   4.3839  6.3448 -3.5212  7.6064  0.0978
#>  -6.7744 -7.3739  0.1667 -4.2430 -5.7810
#> 
#> (1,4,.,.) = 
#>   1.6617   2.8434  -0.9875  -4.1955   0.7345
#>   -4.2787   0.8615   7.5376   2.8767   0.3421
#>    1.4335   1.4242   0.5698   3.3547  -1.1525
#>   -4.5550   3.0767  -1.5104 -10.0451  -1.8274
#>   -2.0916   4.9701   4.8091  -1.7908   1.2392
#> 
#> (1,5,.,.) = 
#>  4.7597 -3.8259 -8.2560 -4.2844  2.1047
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
