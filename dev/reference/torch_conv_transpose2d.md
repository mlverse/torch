# Conv_transpose2d

Conv_transpose2d

## Usage

``` r
torch_conv_transpose2d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padH, padW)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padH, out_padW)`.
  Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

## conv_transpose2d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
inputs = torch_randn(c(1, 4, 5, 5))
weights = torch_randn(c(4, 8, 3, 3))
nnf_conv_transpose2d(inputs, weights, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#>   0.1335  -0.9525  -1.3589  -3.9983  -0.3157
#>    0.1752   4.7588  -4.1934   2.9707  -6.0941
#>    7.3949  -3.8345   5.2146  -8.4070   3.2131
#>   -0.0567   3.7177 -10.7681   3.1401  -4.1882
#>   -5.1038   5.0997  -1.7783   5.2574  -1.6088
#> 
#> (1,2,.,.) = 
#> -1.3054  4.6432 -8.3676 -5.0071 -1.2306
#>   0.1733  1.5175  2.1657  8.2374  2.0519
#>   0.2973  8.3538  1.5651  5.7667 -4.1091
#>   6.1355  4.2480 -8.8398 -3.6326 -3.7606
#>  -1.9662 -0.8167 -0.3212  2.1903  0.1060
#> 
#> (1,3,.,.) = 
#> -0.1479 -1.6130 -4.8676  5.1393  9.9307
#>   3.7804  0.6639 -2.6946 -6.6542  1.4558
#>   9.3187  3.2446 -4.0687  0.5742 -5.1475
#>   6.5794  4.7842 -9.6928  0.4264  0.0839
#>  -6.4152 -0.6637  4.1359  3.0276  5.3291
#> 
#> (1,4,.,.) = 
#>  -0.3828   5.2924  -7.9787  10.2660  -2.0076
#>    6.1314  -4.9304   6.5790  -1.9104   6.1011
#>   -4.6693   7.3197 -13.5142  11.8549 -11.1512
#>    4.7702  -1.2097  -1.0902   5.9161  -1.4167
#>    4.0828  -8.5766   6.2643  -5.6245   0.4162
#> 
#> (1,5,.,.) = 
#>   0.6112  -4.5341 -12.4645  -2.3937   1.1066
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
