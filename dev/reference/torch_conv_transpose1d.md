# Conv_transpose1d

Conv_transpose1d

## Usage

``` r
torch_conv_transpose1d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sW,)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padW,)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padW)`. Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dW,)`. Default: 1

## conv_transpose1d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

inputs = torch_randn(c(20, 16, 50))
weights = torch_randn(c(16, 33, 5))
nnf_conv_transpose1d(inputs, weights)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8   3.4912  -0.8959  -4.7536   0.6242   1.3169   9.2063   9.9684 -10.0838
#>   -1.1077  -2.3907  -6.4638  -4.4825   0.0502   4.5651   1.3244  15.9801
#>   -1.8153  -0.1544   0.8746  -3.4702   1.2023  10.0055  -5.7514  13.3084
#>   -3.1049  -3.6208  -0.3842  -2.5238   1.0048  -4.3974   4.6718 -10.6633
#>   -1.6577   1.9666   2.0100  -8.7428   9.0990   6.9122  -1.5117   7.1916
#>    1.1361  -3.6387  -5.1383  -3.6768  16.5421   5.7267 -13.8310   2.5145
#>    6.0453  -3.7064   2.8052   6.0041  -4.7310  -9.4455   9.7869   0.5474
#>   -3.7006   5.4902  -5.0011   0.7877  10.6058  -4.3896  10.7853  13.6008
#>    1.0146   1.9284  -5.2111   3.4934   0.6224   2.2971  -2.4176 -20.0757
#>   -3.5722 -10.1024   0.5599   8.7532  -8.3165   7.0112  -0.5152  29.1590
#>    3.3903   5.4340  -7.7687   6.6980  -9.4360   3.4321   5.4345  -9.4620
#>   -2.2664   2.6151   5.4524  -8.7638  -4.6629  -4.9237  -0.2379  21.9401
#>    2.1724   0.7995  -1.9336   2.3275  -3.8148  -2.9017  -6.3634  -8.6483
#>   -1.0108  -1.2303   0.8408  -6.3823   1.6991  -2.4079   2.8800   9.4926
#>    3.9834   1.0489  -1.6108  -1.7840  11.6105  -7.8237  17.9496  -8.8330
#>   -1.5647  -2.2678   4.5367  -2.0678  -1.5263   4.0371  -3.0425  -7.7946
#>   -2.6760 -13.7467   5.2979   0.0778  -9.7925  -2.3427 -19.4642   9.3911
#>   -0.4839  -3.5921  -0.7156  -1.2164  -7.8433  -3.0986  17.0038   4.1418
#>   -4.1406  -4.9313   1.5599  10.2275   7.1540  16.8066   0.7057  18.8003
#>    5.2937   5.3270  11.9666  -5.8536  -6.5314 -13.4201   8.1283 -12.6458
#>    0.9368  -1.4967   7.9342   2.6670 -10.6289  -6.7362  11.0528  -7.0961
#>   -0.1214 -10.8997   2.2915  -0.2508   5.0787  14.2098  -3.1802  -8.6145
#>   -0.0924   3.1744  -2.5435  -9.7792   2.8743  -5.5009  -5.6751   2.4547
#>   -4.9433 -15.0998  12.6870   0.5146  -1.4111  -0.1358   0.1406  -2.6702
#>   -1.8125  -0.7010  -3.5130  -8.7360 -13.9368   3.3701 -15.3282   4.2187
#>   -5.6727  -3.8722  -2.3514  -0.2362   1.0902  -1.2238   3.0933   6.9658
#>    1.3892   0.6092  10.0502   0.2712  -7.8285  -4.1636  -2.8720   3.5144
#>   -2.6828   4.1502   3.3710   5.7034  16.3476 -16.0907   4.9140   4.8271
#>   -0.7435  -1.6594  -1.6269  -8.6778   7.4385  -2.4245  17.9695 -18.2728
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
