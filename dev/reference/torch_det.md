# Det

Det

## Usage

``` r
torch_det(self)
```

## Arguments

- self:

  (Tensor) the input tensor of size `(*, n, n)` where `*` is zero or
  more batch dimensions.

## Note

    Backward through `det` internally uses SVD results when `input` is
    not invertible. In this case, double backward through `det` will be
    unstable in when `input` doesn't have distinct singular values. See
    `~torch.svd` for details.

## det(input) -\> Tensor

Calculates determinant of a square matrix or batches of square matrices.

## Examples

``` r
if (torch_is_installed()) {

A = torch_randn(c(3, 3))
torch_det(A)
A = torch_randn(c(3, 2, 2))
A
A$det()
}
#> torch_tensor
#> -1.6040
#>  1.1829
#> -0.1495
#> [ CPUFloatType{3} ]
```
