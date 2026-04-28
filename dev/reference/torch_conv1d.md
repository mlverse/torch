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
#> Columns 1 to 6  4.2632e-01  4.8286e+00  6.5760e+00 -7.4070e+00  2.9554e+00  6.5482e+00
#>  -4.7483e+00  3.9550e+00  3.8306e+00 -1.0715e+01 -2.0047e+00  9.5566e+00
#>   2.5558e+00 -7.7854e+00  4.7738e+00  7.5902e+00 -2.1939e+00  2.6704e+00
#>  -1.0667e+01 -6.8660e+00  1.2278e+00 -3.6622e+00  9.0553e+00 -1.0721e+01
#>  -9.4726e+00  5.5603e+00  1.0261e+00 -2.6837e+00 -2.5263e+00  5.9090e-01
#>  -5.9544e+00 -3.2626e+00  9.3766e+00 -8.2326e+00  2.7267e+00 -2.6626e+00
#>   4.9968e+00  2.9819e+00  9.0928e-01 -7.6399e+00  7.7726e+00 -1.3400e+01
#>  -1.6389e+00  6.0553e-01 -4.5445e+00  9.4712e-01 -1.6349e+00  3.0190e+00
#>   5.3974e+00 -2.6172e+00 -2.9068e-01  6.2100e+00 -4.2269e+00  7.7827e-01
#>  -3.0198e+00 -1.3111e+01 -6.7593e+00 -2.3478e+00  4.2989e+00  5.8847e+00
#>  -1.5204e+00  3.3154e+00  5.2033e+00 -9.1925e+00 -3.6624e-01  2.3225e+00
#>   2.0953e-01 -1.0353e+00  5.7540e+00  1.0315e+00  3.5268e+00 -8.5381e-01
#>   8.1971e+00 -7.6027e+00  2.1078e+00  1.6199e+00  4.5944e+00  5.0890e+00
#>   1.8780e+00  6.9528e+00  4.5902e+00 -1.6020e+00  2.9407e-01  7.0651e+00
#>   1.8850e-01  8.6511e-02  1.2628e+00 -1.1216e+01  5.5259e+00 -7.0278e+00
#>   2.6950e+00 -2.6245e+00 -1.1888e+01  1.3481e+00  1.4977e+01 -8.2877e+00
#>  -9.6916e-01 -2.9483e+00 -7.0423e+00 -4.3263e+00  5.2511e+00 -6.8992e+00
#>   7.3627e-01 -2.9899e+00  3.1634e+00  7.3620e+00 -2.0780e+00  8.4801e-01
#>  -1.3773e+01 -8.0740e+00  7.6772e+00  4.9076e+00 -7.5482e-01  4.9525e+00
#>  -2.9746e+00 -7.4355e-01  3.1380e+00  6.9238e-01  3.6811e+00 -4.7354e+00
#>  -5.9966e+00  5.4518e+00  5.9502e+00 -2.5341e+00 -2.5820e+00 -7.0070e+00
#>   5.9414e+00 -3.8787e+00  8.0571e+00  2.8486e+00  1.0361e+01  3.6980e+00
#>  -1.4386e+01 -3.4376e+00 -4.4591e+00  8.8327e+00  9.2142e+00  2.7345e-01
#>   1.5206e+01 -3.3159e-01 -1.1385e+01  2.2386e+00  7.4046e+00  2.3756e+00
#>   7.1784e+00  1.3915e+00  1.1179e+00  3.2872e+00 -1.2442e+00  1.8538e+00
#>   3.7012e+00 -4.7291e+00 -1.1353e+00 -1.0379e+00 -9.0182e-01  5.5841e+00
#>   5.9960e+00  4.2324e-01 -9.4853e+00  3.4498e+00  1.2833e+00  5.2983e-05
#>  -9.9490e+00 -8.1723e+00 -5.3398e+00  9.3012e-02  8.2266e+00 -4.6359e+00
#>  -1.2695e+01  3.7109e+00  8.9099e+00 -5.3721e+00  4.7289e+00 -1.1756e+01
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
