# Logical_or

Logical_or

## Usage

``` r
torch_logical_or(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the tensor to compute OR with

## logical_or(input, other, out=NULL) -\> Tensor

Computes the element-wise logical OR of the given input tensors. Zeros
are treated as `FALSE` and nonzeros are treated as `TRUE`.

## Examples

``` r
if (torch_is_installed()) {

torch_logical_or(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
torch_logical_or(a, b)
if (FALSE) { # \dontrun{
torch_logical_or(a$double(), b$double())
torch_logical_or(a$double(), b)
torch_logical_or(a, b, out=torch_empty(4, dtype=torch_bool()))
} # }
}
```
