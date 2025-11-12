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
#>  Columns 1 to 8   0.8243  -3.5578  -0.4207  -3.7491  -2.8253   3.9587  10.5372 -12.0174
#>    6.0397  -0.0181  -4.9057   9.9906  -1.2323  11.8297 -15.7933  -2.1774
#>    0.0796   0.0744   2.3972   0.8247   9.3287   5.5408   6.8007  -4.3315
#>   -4.1222  -0.0645   4.6291   3.5720  -1.2406  -2.6391   1.6905  -4.3403
#>   -4.3540  -2.9915  -4.1605   0.9453   7.7003  -5.0640  -3.9178   1.6225
#>   -0.3340   3.9311  -2.8927 -14.3713  12.1778  31.5201 -13.8543  -6.8353
#>    2.3241   6.2071   9.4884   8.6996  -4.7185 -11.4582  -5.3514 -10.0118
#>    1.3083  -4.0927   5.2914   3.6070  -0.3234  -4.7311  -1.6133  13.8588
#>   -4.6592  -3.2322   9.9715   1.2603  -8.1246  -4.8882  -0.0134   5.6275
#>   -1.3503  13.3650  10.1084   1.0622  -4.4238   2.4426  -5.2360 -15.9605
#>   -4.7170  -2.8533   2.2213   4.2660   5.7695   6.9929   0.3901   0.4961
#>  -11.1714  -4.5926  -1.8171   4.3238  15.0333   5.6772   9.8975  11.3683
#>   -0.1048  -6.4937   1.8557 -13.7931  -5.9877  -8.0427   5.1385   8.9329
#>   -4.6457  -1.2212  -6.6208   2.2354  -0.3885  -0.1190  11.1541  -6.0244
#>   -0.2252   5.4345   0.5722  -0.5158  -3.8952  -8.3624   4.8296  12.6656
#>   -7.3323   3.9670  -1.1668  13.3687  -1.7070 -10.8230 -15.9144  -8.0842
#>    1.2299  12.3053  -6.8043   1.9872  -6.4942  -0.3874 -15.5778  10.2735
#>   -0.7025 -18.8140  -4.1338  10.7963   3.7480  19.8133   3.7222  -9.3884
#>   -1.5539  -1.3034   7.0027   9.1300   1.6345   9.4519   2.3426   4.2045
#>    5.1011  -2.5824  -9.6812   0.1660   5.4078  -2.6380   9.4704 -14.6490
#>    1.3179  -0.1102  -3.4215   5.4656  17.0802   3.8890 -11.9647  -2.3237
#>   -0.7291   2.2443   3.0752   3.6788 -10.5572  -2.9748   0.8562   9.3081
#>    1.2385  -5.0241   7.6051   4.8673  -4.8362  17.4458 -12.9930  -8.3460
#>    6.5020  -0.1705  -8.7482  11.7890   1.5506  15.2942  -5.5519 -10.1429
#>  -10.3326   4.2948  -8.1279 -10.9383  -8.1169  -6.8552   2.1908   4.8804
#>   -6.1246   4.2120  -2.2420  -2.9700   3.5491   0.1291  -7.8972  -0.7472
#>   -2.1725   6.2845   3.4125  -4.1876  -9.9464  -2.0724  -4.3607  -0.8291
#>    4.9537   6.1075   0.0855 -12.3396  -2.7965  -1.1515   7.7249  -6.8396
#>    1.1562   5.1133  -7.8473  -0.2288  -4.1200  -3.2510  -3.3658   0.4280
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
