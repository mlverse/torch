# Isreal

Isreal

## Usage

``` r
torch_isreal(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## isreal(input) -\> Tensor

Returns a new tensor with boolean elements representing if each element
of `input` is real-valued or not. All real-valued types are considered
real. Complex values are considered real when their imaginary part is 0.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) {
torch_isreal(torch_tensor(c(1, 1+1i, 2+0i)))
}
}
```
