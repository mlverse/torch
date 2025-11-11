# Pinverse

Pinverse

## Usage

``` r
torch_pinverse(self, rcond = 1e-15)
```

## Arguments

- self:

  (Tensor) The input tensor of size \\(\*, m, n)\\ where \\\*\\ is zero
  or more batch dimensions

- rcond:

  (float) A floating point value to determine the cutoff for small
  singular values. Default: 1e-15

## Note

    This method is implemented using the Singular Value Decomposition.

    The pseudo-inverse is not necessarily a continuous function in the elements of the matrix `[1]`_.
    Therefore, derivatives are not always existent, and exist for a constant rank only `[2]`_.
    However, this method is backprop-able due to the implementation by using SVD results, and
    could be unstable. Double-backward will also be unstable due to the usage of SVD internally.
    See `~torch.svd` for more details.

## pinverse(input, rcond=1e-15) -\> Tensor

Calculates the pseudo-inverse (also known as the Moore-Penrose inverse)
of a 2D tensor. Please look at `Moore-Penrose inverse`\_ for more
details

## Examples

``` r
if (torch_is_installed()) {

input = torch_randn(c(3, 5))
input
torch_pinverse(input)
# Batched pinverse example
a = torch_randn(c(2,6,3))
b = torch_pinverse(a)
torch_matmul(b, a)
}
#> torch_tensor
#> (1,.,.) = 
#>   1.0000e+00  3.6728e-07 -2.4137e-07
#>   5.8201e-08  1.0000e+00 -7.9490e-08
#>  -6.0175e-08  2.1168e-07  1.0000e+00
#> 
#> (2,.,.) = 
#>   1.0000e+00 -3.5376e-07  7.0673e-07
#>  -2.0992e-08  1.0000e+00 -1.7638e-07
#>   4.2397e-09 -8.2702e-08  1.0000e+00
#> [ CPUFloatType{2,3,3} ]
```
