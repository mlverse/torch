# Logdet

Logdet

## Usage

``` r
torch_logdet(self)
```

## Arguments

- self:

  (Tensor) the input tensor of size `(*, n, n)` where `*` is zero or
  more batch dimensions.

## Note

    Result is `-inf` if `input` has zero log determinant, and is `NaN` if
    `input` has negative determinant.

    Backward through `logdet` internally uses SVD results when `input`
    is not invertible. In this case, double backward through `logdet` will
    be unstable in when `input` doesn't have distinct singular values. See
    `~torch.svd` for details.

## logdet(input) -\> Tensor

Calculates log determinant of a square matrix or batches of square
matrices.

## Examples

``` r
if (torch_is_installed()) {

A = torch_randn(c(3, 3))
torch_det(A)
torch_logdet(A)
A
A$det()
A$det()$log()
}
#> torch_tensor
#> 0.548664
#> [ CPUFloatType{} ]
```
