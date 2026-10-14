# Logical_xor

Logical_xor

## Usage

``` r
torch_logical_xor(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the tensor to compute XOR with

## logical_xor(input, other, out=NULL) -\> Tensor

Computes the element-wise logical XOR of the given input tensors. Zeros
are treated as `FALSE` and nonzeros are treated as `TRUE`.

## Examples

``` r
if (torch_is_installed()) {

torch_logical_xor(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
torch_logical_xor(a, b)
torch_logical_xor(a$to(dtype=torch_double()), b$to(dtype=torch_double()))
torch_logical_xor(a$to(dtype=torch_double()), b)
}
#> torch_tensor
#>  1
#>  1
#>  0
#>  0
#> [ CPUBoolType{4} ]
```
