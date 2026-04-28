# Mm

Mm

## Usage

``` r
torch_mm(self, mat2, out_dtype)
```

## Arguments

- self:

  (Tensor) the first matrix to be multiplied

- mat2:

  (Tensor) the second matrix to be multiplied

- out_dtype:

  (torch_dtype, optional) the output dtype

## Note

This function does not broadcast . For broadcasting matrix products, see
[`torch_matmul`](https://torch.mlverse.org/docs/dev/reference/torch_matmul.md).

## mm(input, mat2, out=NULL) -\> Tensor

Performs a matrix multiplication of the matrices `input` and `mat2`.

If `input` is a \\(n \times m)\\ tensor, `mat2` is a \\(m \times p)\\
tensor, `out` will be a \\(n \times p)\\ tensor.

## Examples

``` r
if (torch_is_installed()) {

mat1 = torch_randn(c(2, 3))
mat2 = torch_randn(c(3, 3))
torch_mm(mat1, mat2)
}
#> torch_tensor
#>  1.7130  0.6111 -0.1842
#>  5.6364  0.8906  0.6283
#> [ CPUFloatType{2,3} ]
```
