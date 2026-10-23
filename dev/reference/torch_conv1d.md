# Conv1d

Conv1d

## Usage

``` r
torch_conv1d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv1d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See
[`nn_conv1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

filters = torch_randn(c(33, 16, 3))
inputs = torch_randn(c(20, 16, 50))
nnf_conv1d(inputs, filters)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 6  5.6410e+00  5.5863e+00 -1.9646e+00  1.2330e+00  5.8130e+00 -1.8004e-01
#>   8.4991e+00 -5.1213e+00 -3.2736e+00  1.0206e+00 -8.2140e-02  1.0776e+00
#>   5.0759e+00 -4.0472e+00  1.2101e+01  5.9956e+00  7.2154e+00 -6.8575e-01
#>  -4.7366e+00  9.9022e+00  1.2873e+00  1.0014e+01 -1.2465e-01 -7.9159e+00
#>   2.0594e+01 -9.1819e-01 -1.0670e+01  3.8808e+00 -4.2788e+00  7.6654e+00
#>  -5.1791e+00 -1.8713e+00 -3.7447e+00 -7.0560e-02 -2.3167e+00 -9.3630e+00
#>  -3.1051e+00  3.3840e-01  5.6237e+00 -5.4586e+00 -5.2067e+00  4.2122e+00
#>   9.3340e+00  2.4514e+00 -2.0388e+01  9.8540e+00  6.5266e+00 -1.4354e+00
#>  -8.3221e+00  1.7085e+01 -1.0347e+01 -1.1124e+01  2.0243e+01  2.3293e+00
#>  -1.9556e+01  4.6225e-01 -8.1540e+00 -6.8692e+00 -5.3452e+00  9.1522e+00
#>  -4.7577e+00 -7.6653e+00  1.6325e-01  1.1477e-01 -3.9706e+00  1.1132e+01
#>  -2.5421e+01  7.4989e+00 -1.1308e+01  9.3959e+00  5.3522e+00 -3.1418e+00
#>   1.8612e+01  6.3234e+00 -4.3296e+00 -1.0089e+00  1.1242e+00  1.0214e+01
#>   1.1744e+00  1.3235e+01 -2.8306e+00 -7.9106e+00  5.1259e+00 -6.4993e+00
#>   9.0393e+00 -9.7249e+00  9.1714e+00  5.4372e-01  1.2077e+01 -2.0801e+00
#>  -2.4308e+00 -4.0607e+00 -5.0455e+00  5.9020e+00  4.7658e+00  4.0731e+00
#>  -1.4582e-01 -1.0183e+01 -1.0092e+01 -1.0487e+01  4.4970e+00  3.0656e+00
#>   1.0966e+01 -7.8333e-01 -1.1535e+01  8.3940e-01 -7.7899e+00 -1.3464e+01
#>  -8.0270e+00 -4.8648e+00  1.4177e+00 -5.7610e+00  3.1124e+00  1.6833e+00
#>   2.9277e+00  1.4196e+00 -2.4658e+00  4.1964e+00  5.8375e+00 -8.3201e+00
#>   9.9047e+00  8.8294e-01 -4.1468e+00 -9.4809e-01  4.8061e+00  2.7927e+00
#>  -6.3559e+00 -6.0209e-01 -5.5430e+00 -1.4697e+01  7.5478e+00  1.6246e+01
#>   7.2862e+00 -1.5515e+01 -4.5369e-01 -5.2136e+00  1.0250e+00  1.3536e+00
#>  -2.9680e+00 -3.5932e+00  1.1242e+01  1.1983e+01 -6.2493e+00 -7.6693e+00
#>   1.3365e+01  3.9729e+00 -1.2054e+01  2.3563e+00  4.6519e+00 -9.5141e-01
#>  -4.2814e+00  8.9544e+00  3.7912e+00  8.6785e+00 -1.0902e+01 -8.1407e+00
#>  -3.8502e+00 -1.3884e+01 -6.3079e-01  2.5803e+00  2.2780e+00  9.7747e+00
#>   6.7117e-01 -2.1034e+01 -3.7880e+00  6.9202e+00  8.8095e+00  5.2751e+00
#>   5.2505e+00  3.7340e+00 -1.9200e+00  9.3776e-01  3.9410e+00  3.6196e-01
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
