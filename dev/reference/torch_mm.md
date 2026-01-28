# Mm

Mm

## Usage

``` r
torch_mm(self, mat2)
```

## Arguments

- self:

  (Tensor) the first matrix to be multiplied

- mat2:

  (Tensor) the second matrix to be multiplied

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
#> -0.5476 -0.1695  0.5091
#> -1.4120 -5.5113 -3.7926
#> [ CPUFloatType{2,3} ]
```
