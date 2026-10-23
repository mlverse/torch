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
#> Columns 1 to 6  1.3489e+00 -1.0436e+01  8.8854e+00  6.5728e+00 -1.3287e+01  1.8465e+01
#>   6.0470e+00 -6.7021e+00 -4.4437e-01 -1.0046e+01  1.5062e+01  5.5014e-01
#>   6.0208e+00  4.7654e+00 -5.3190e+00 -4.5118e+00 -6.2624e+00  1.1082e+01
#>  -1.1975e+00  1.8011e+00  1.5705e+00 -1.4881e+00 -4.2890e+00  5.4680e-01
#>   1.0798e+01 -1.5051e+00  3.8358e+00  1.3339e+00  3.9798e+00 -5.1435e-01
#>   3.8861e+00  5.6392e-01  6.7872e+00 -1.0727e+01  3.9607e+00 -6.0261e+00
#>   3.2634e-01  1.0348e+01 -7.8149e+00  3.7198e+00 -1.2961e+01  1.7799e+00
#>  -4.5165e+00  5.5456e+00  5.9515e+00 -4.2474e+00 -4.0311e+00  1.2387e+01
#>  -5.8350e+00 -8.7138e+00  4.4447e+00  1.2151e+01  1.9384e+00 -3.5528e+00
#>  -4.2025e+00  1.2365e+01 -3.0780e+00 -3.1027e+00  1.5460e+00  1.4818e+01
#>   4.2176e+00 -5.0511e+00 -2.6281e+00  1.1132e+01  2.0037e+00  1.6046e+01
#>   2.3559e+00 -3.4136e+00  5.2185e-01  3.4622e-01  5.2542e+00  4.4881e+00
#>  -1.3512e+01 -1.5095e+00 -3.5114e-01 -4.0324e+00 -4.9655e-02  7.6812e+00
#>   6.0509e+00 -5.4711e-01 -3.3036e+00 -4.7946e+00  4.9163e-01  9.6602e+00
#>  -1.9244e+00 -7.3882e+00 -1.3993e+00 -6.7338e+00  9.7039e+00  1.1296e+00
#>  -1.1860e+00 -5.5293e+00  1.0200e+01  1.1447e+01 -2.5624e+00 -7.4820e-01
#>  -6.3233e+00 -3.6336e+00 -1.5939e+00  1.0335e+00  7.2104e+00 -2.1445e-01
#>  -3.7690e+00  2.4206e+00 -4.2047e-01 -1.7575e+00 -5.6088e+00  5.0212e+00
#>   3.6734e+00  3.1707e+00 -1.7159e+00 -7.6395e+00  4.7128e+00  5.3627e+00
#>  -6.2373e-01  5.7181e+00 -8.4996e+00  2.4876e+00 -3.2049e+00 -1.2732e+01
#>  -2.4855e+00 -1.2811e+01 -2.4105e+00  8.6082e+00  9.6139e+00  5.8293e+00
#>   1.2000e+00 -1.1917e+00  6.4281e+00  8.4604e+00 -6.2386e+00 -1.6497e+01
#>  -7.5707e-02  1.0641e+01  3.7091e+00 -1.4441e+00  5.8574e+00  3.1452e+00
#>   6.6122e+00 -3.2475e+00  7.5530e+00  5.2467e+00 -6.2518e+00  1.4308e-01
#>  -1.3087e+01  5.2585e+00 -3.9805e+00  7.5039e-01 -5.9216e+00  1.9000e+01
#>  -1.6580e+01  1.2754e+01 -5.6690e+00  7.2381e+00  1.2439e+01 -1.0784e+00
#>  -2.8053e+00  6.8964e+00  1.5731e+00  4.0868e+00  4.0165e-01  1.0170e+01
#>  -9.2700e+00  3.3521e+00  5.6745e-01  1.3591e+00 -2.2150e+00  2.0275e+01
#>   1.7932e+01 -1.6135e+00  4.4999e-01 -1.0651e+01  3.4038e+00 -8.0326e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
