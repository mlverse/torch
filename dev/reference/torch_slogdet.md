# Slogdet

Slogdet

## Usage

``` r
torch_slogdet(self)
```

## Arguments

- self:

  (Tensor) the input tensor of size `(*, n, n)` where `*` is zero or
  more batch dimensions.

## Note

    If `input` has zero determinant, this returns `(0, -inf)`.

    Backward through `slogdet` internally uses SVD results when `input`
    is not invertible. In this case, double backward through `slogdet`
    will be unstable in when `input` doesn't have distinct singular values.
    See `~torch.svd` for details.

## slogdet(input) -\> (Tensor, Tensor)

Calculates the sign and log absolute value of the determinant(s) of a
square matrix or batches of square matrices.

## Examples

``` r
if (torch_is_installed()) {

A = torch_randn(c(3, 3))
A
torch_det(A)
torch_logdet(A)
torch_slogdet(A)
}
#> [[1]]
#> torch_tensor
#> 1
#> [ CPUFloatType{} ]
#> 
#> [[2]]
#> torch_tensor
#> 0.773673
#> [ CPUFloatType{} ]
#> 
```
