# Inverse

Inverse

## Usage

``` r
torch_inverse(self)
```

## Arguments

- self:

  (Tensor) the input tensor of size \\(\*, n, n)\\ where `*` is zero or
  more batch dimensions

## Note

    Irrespective of the original strides, the returned tensors will be
    transposed, i.e. with strides like `input.contiguous().transpose(-2, -1).stride()`

## inverse(input, out=NULL) -\> Tensor

Takes the inverse of the square matrix `input`. `input` can be batches
of 2D square tensors, in which case this function would return a tensor
composed of individual inverses.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
x = torch_rand(c(4, 4))
y = torch_inverse(x)
z = torch_mm(x, y)
z
torch_max(torch_abs(z - torch_eye(4))) # Max non-zero
# Batched inverse example
x = torch_randn(c(2, 3, 4, 4))
y = torch_inverse(x)
z = torch_matmul(x, y)
torch_max(torch_abs(z - torch_eye(4)$expand_as(x))) # Max non-zero
} # }
}
```
