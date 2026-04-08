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
#>  -0.5789  -3.8957   5.4122 -12.2709   3.5351
#>    1.2619   1.1721 -12.8529  11.5754   5.3082
#>   -0.1034   4.5266  -0.4010  -8.0670   0.4009
#>   -6.3384  -4.6164   8.8227   5.0094  -9.9373
#>   -0.5042  -0.9244  -1.8482   3.2413   2.9443
#> 
#> (1,2,.,.) = 
#> -0.1181 -3.5908  7.8447  4.1223  0.4336
#>   4.6174  1.7512  3.7262  3.8882 -6.1433
#>   4.3041  0.1862 -4.9174 -3.0050  0.5348
#>   4.6723 -5.1084 -3.5815  3.8556  1.8165
#>  -2.1625 -3.4852  3.3161 -1.3207 -5.9894
#> 
#> (1,3,.,.) = 
#>  -2.9604  -2.9096  -1.7728   6.3700  -6.1445
#>    7.6661  -4.7580  -2.7041   4.2738  -5.8233
#>   -1.8479  -1.4278  -2.3742  -3.0445   6.9120
#>   -4.0984  -5.7618  12.4913   5.0548  -9.4680
#>   -5.7196   2.1022  -3.6373   4.4772   2.9763
#> 
#> (1,4,.,.) = 
#>  -6.5965  -2.7961   5.1601   0.1688  12.6899
#>   -6.5131   5.3626  11.0222 -16.9024   9.1361
#>   -4.8203   3.5252  -3.9381  -7.1324   9.4248
#>    2.5408   5.0933  -8.6994  -5.9905   3.8383
#>    4.1814  -2.4827   4.6082  -2.3855  -5.0628
#> 
#> (1,5,.,.) = 
#>   4.7463   9.1660  -1.8983   9.9004  -9.1623
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
