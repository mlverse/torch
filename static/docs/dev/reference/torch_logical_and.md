# Logical_and

Logical_and

## Usage

``` r
torch_logical_and(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the tensor to compute AND with

## logical_and(input, other, out=NULL) -\> Tensor

Computes the element-wise logical AND of the given input tensors. Zeros
are treated as `FALSE` and nonzeros are treated as `TRUE`.

## Examples

``` r
if (torch_is_installed()) {

torch_logical_and(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
torch_logical_and(a, b)
if (FALSE) { # \dontrun{
torch_logical_and(a, b, out=torch_empty(4, dtype=torch_bool()))
} # }
}
```
