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
#>  1.0000e+00  1.0956e-07 -4.7671e-08
#>   1.3874e-08  1.0000e+00 -5.3458e-10
#>  -6.2668e-09 -2.9059e-08  1.0000e+00
#> 
#> (2,.,.) = 
#>  1.0000e+00  4.2986e-08 -3.9264e-08
#>  -8.6679e-08  1.0000e+00  1.2691e-07
#>  -1.3235e-08 -1.2572e-07  1.0000e+00
#> [ CPUFloatType{2,3,3} ]
```
