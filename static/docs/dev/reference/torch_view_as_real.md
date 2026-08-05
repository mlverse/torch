# View_as_real

View_as_real

## Usage

``` r
torch_view_as_real(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## view_as_real(input) -\> Tensor

Returns a view of `input` as a real tensor. For an input complex tensor
of `size` \\m1, m2, \dots, mi\\, this function returns a new real tensor
of size \\m1, m2, \dots, mi, 2\\, where the last dimension of size 2
represents the real and imaginary components of complex numbers.

## Warning

`torch_view_as_real()` is only supported for tensors with
`complex dtypes`.

## Examples

``` r
if (torch_is_installed()) {

if (FALSE) {
x <- torch_randn(4, dtype=torch_cfloat())
x
torch_view_as_real(x)
}
}
```
