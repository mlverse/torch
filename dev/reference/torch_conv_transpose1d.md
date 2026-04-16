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
#> Columns 1 to 8  11.6797  -3.6015   9.6837   5.2433  19.6952   9.6237  -9.1988   4.7676
#>   -2.6132   5.1373  -0.1230  -3.4059   6.4851 -14.4240   4.1904   6.9861
#>   -7.6093  -6.9915  -2.0172  -2.9472 -16.8709   0.8575  11.2457 -20.8546
#>    1.5468  -3.8014  -8.7607   0.5537  -1.0214 -11.1687  12.6668 -13.2721
#>   -7.5031  -1.9433  -5.2318  -0.4211  -8.0136  -0.3645   0.3339  -3.5401
#>    0.5942  -0.2639  19.7645  10.4237  -8.9767  -7.1139   7.3266   2.9040
#>   -8.7123  -4.0566  -2.3466  -0.0259   5.3601   6.1435   3.7513  -2.4091
#>    2.1298  -8.2629  11.3592   2.6324 -11.7159   3.0418   4.5778   4.5424
#>    6.1748  -0.2367  -8.9182  10.8340   7.8844  -8.5392 -15.1708  -5.8959
#>    2.3323  10.5159  -9.4111  -7.3605  -4.1868  -1.3200  -0.0826   1.0757
#>   12.4698  -3.3256  -3.9746  18.7472 -18.6355   1.4814   4.0559 -14.8201
#>    8.5983  -0.7965   2.9529   9.6844   0.2635   4.2486   1.2459  -9.8783
#>    1.0927  -2.9577   0.2076  -2.6434   2.2039   8.3389  11.3815  22.2479
#>   -5.3479   0.2424   2.8941 -12.8289 -26.4989  13.4263  -8.2476 -15.0234
#>   -1.7676  -4.9578  12.8724  -0.0305  -4.5133  -3.5969  13.6589 -10.5064
#>    4.3454  -1.5868   8.4317  -5.9588 -14.6713   1.6427  -6.1520 -14.5397
#>    2.0268   4.5852   6.3074  -6.9091  15.4538  -3.2767  -6.0499   3.0283
#>    3.2540   7.6323  -2.6481  -5.3085   6.9912   8.7958  11.4542   1.4885
#>    1.0293 -11.3447  17.2668 -13.1404  -6.4917   0.9984 -16.8567   6.9678
#>    2.8199   8.6114  -3.9635 -12.5743   0.2054  -4.5949  11.3289  -6.4532
#>    1.9995   0.4349  -5.7067   0.5553   5.0301  -1.3124  -2.9513  -8.3293
#>   10.4700  -1.1727   4.1165  17.4050  -8.6369   3.5860  -0.2454  -2.7965
#>   -7.2551  11.1835 -16.2529 -10.7449   8.7475 -11.3090   5.7577   2.1755
#>   -5.0507  -6.4363  12.4727  -9.5515   3.4992   2.4633  -2.9127   1.2971
#>   -3.9283   1.7952   1.2129  13.6154  18.5474   1.9070   0.8735   0.7658
#>  -10.0614   2.5726  -1.0081  -7.4275   1.7230 -14.1397  -1.1472  -0.9290
#>   -5.1266  18.2435   0.4453  -5.9433  -5.9596  -7.9317   6.9958 -10.7208
#>   -8.6052  10.1144   3.1995   5.4874  11.7402   8.1279 -14.8996  -1.3906
#>    3.8305   1.2193   2.2302 -11.5182 -17.4353   2.1050   3.8393  -4.3532
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
