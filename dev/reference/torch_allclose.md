# Allclose

Allclose

## Usage

``` r
torch_allclose(self, other, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE)
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

  (bool, optional) if `TRUE`, then two `NaN` s will be compared as
  equal. Default: `FALSE`

## allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -\> bool

This function checks if all `input` and `other` satisfy the condition:

\$\$ \vert \mbox{input} - \mbox{other} \vert \leq \mbox{atol} +
\mbox{rtol} \times \vert \mbox{other} \vert \$\$ elementwise, for all
elements of `input` and `other`. The behaviour of this function is
analogous to
`numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`\_

## Examples

``` r
if (torch_is_installed()) {

torch_allclose(torch_tensor(c(10000., 1e-07)), torch_tensor(c(10000.1, 1e-08)))
torch_allclose(torch_tensor(c(10000., 1e-08)), torch_tensor(c(10000.1, 1e-09)))
torch_allclose(torch_tensor(c(1.0, NaN)), torch_tensor(c(1.0, NaN)))
torch_allclose(torch_tensor(c(1.0, NaN)), torch_tensor(c(1.0, NaN)), equal_nan=TRUE)
}
#> [1] TRUE
```
