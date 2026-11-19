# View_as_complex

View_as_complex

## Usage

``` r
torch_view_as_complex(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## view_as_complex(input) -\> Tensor

Returns a view of `input` as a complex tensor. For an input complex
tensor of `size` \\m1, m2, \dots, mi, 2\\, this function returns a new
complex tensor of `size` \\m1, m2, \dots, mi\\ where the last dimension
of the input tensor is expected to represent the real and imaginary
components of complex numbers.

## Warning

torch_view_as_complex is only supported for tensors with `torch_dtype`
[`torch_float64()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md)
and
[`torch_float32()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md).
The input is expected to have the last dimension of `size` 2. In
addition, the tensor must have a `stride` of 1 for its last dimension.
The strides of all other dimensions must be even numbers.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) {
x=torch_randn(c(4, 2))
x
torch_view_as_complex(x)
}
}
```
