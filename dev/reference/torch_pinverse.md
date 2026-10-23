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
#>  1.0000e+00  2.5841e-11  1.4645e-07
#>   8.6423e-08  1.0000e+00 -1.1243e-07
#>   1.1583e-07  5.3680e-08  1.0000e+00
#> 
#> (2,.,.) = 
#>  1.0000e+00 -6.4514e-08  4.0298e-08
#>   9.8450e-09  1.0000e+00 -6.8330e-08
#>  -2.7392e-08  5.4587e-08  1.0000e+00
#> [ CPUFloatType{2,3,3} ]
```
