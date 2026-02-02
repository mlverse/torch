# Isclose

Isclose

## Usage

``` r
torch_isclose(self, other, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE)
```

## Arguments

- self:

  (Tensor) first tensor to compare

- other:

  (Tensor) second tensor to compare

- rtol:

  (float, optional) relative tolerance. Default: 1e-05

- atol:

  (float, optional) absolute tolerance. Default: 1e-08

- equal_nan:

  (bool, optional) if `TRUE`, then two `NaN` s will be considered equal.
  Default: `FALSE`

## isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=FALSE) -\> Tensor

Returns a new tensor with boolean elements representing if each element
of `input` is "close" to the corresponding element of `other`. Closeness
is defined as:

\$\$ \vert \mbox{input} - \mbox{other} \vert \leq \mbox{atol} +
\mbox{rtol} \times \vert \mbox{other} \vert \$\$

where `input` and `other` are finite. Where `input` and/or `other` are
nonfinite they are close if and only if they are equal, with NaNs being
considered equal to each other when `equal_nan` is TRUE.

## Examples

``` r
if (torch_is_installed()) {

torch_isclose(torch_tensor(c(1., 2, 3)), torch_tensor(c(1 + 1e-10, 3, 4)))
torch_isclose(torch_tensor(c(Inf, 4)), torch_tensor(c(Inf, 6)), rtol=.5)
}
#> torch_tensor
#>  1
#>  1
#> [ CPUBoolType{2} ]
```
