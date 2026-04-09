# Matmul

Matmul

## Usage

``` r
torch_matmul(self, other)
```

## Arguments

- self:

  (Tensor) the first tensor to be multiplied

- other:

  (Tensor) the second tensor to be multiplied

## Note

    The 1-dimensional dot product version of this function does not support an `out` parameter.

## matmul(input, other, out=NULL) -\> Tensor

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

- If both tensors are 1-dimensional, the dot product (scalar) is
  returned.

- If both arguments are 2-dimensional, the matrix-matrix product is
  returned.

- If the first argument is 1-dimensional and the second argument is
  2-dimensional, a 1 is prepended to its dimension for the purpose of
  the matrix multiply. After the matrix multiply, the prepended
  dimension is removed.

- If the first argument is 2-dimensional and the second argument is
  1-dimensional, the matrix-vector product is returned.

- If both arguments are at least 1-dimensional and at least one argument
  is N-dimensional (where N \> 2), then a batched matrix multiply is
  returned. If the first argument is 1-dimensional, a 1 is prepended to
  its dimension for the purpose of the batched matrix multiply and
  removed after. If the second argument is 1-dimensional, a 1 is
  appended to its dimension for the purpose of the batched matrix
  multiple and removed after. The non-matrix (i.e. batch) dimensions are
  broadcasted (and thus must be broadcastable). For example, if `input`
  is a \\(j \times 1 \times n \times m)\\ tensor and `other` is a \\(k
  \times m \times p)\\ tensor, `out` will be an \\(j \times k \times n
  \times p)\\ tensor.

## Examples

``` r
if (torch_is_installed()) {

# vector x vector
tensor1 = torch_randn(c(3))
tensor2 = torch_randn(c(3))
torch_matmul(tensor1, tensor2)
# matrix x vector
tensor1 = torch_randn(c(3, 4))
tensor2 = torch_randn(c(4))
torch_matmul(tensor1, tensor2)
# batched matrix x broadcasted vector
tensor1 = torch_randn(c(10, 3, 4))
tensor2 = torch_randn(c(4))
torch_matmul(tensor1, tensor2)
# batched matrix x batched matrix
tensor1 = torch_randn(c(10, 3, 4))
tensor2 = torch_randn(c(10, 4, 5))
torch_matmul(tensor1, tensor2)
# batched matrix x broadcasted matrix
tensor1 = torch_randn(c(10, 3, 4))
tensor2 = torch_randn(c(4, 5))
torch_matmul(tensor1, tensor2)
}
#> torch_tensor
#> (1,.,.) = 
#>  1.4269  1.6369  0.7131  0.1074 -1.5977
#>  -2.8428  1.7875  3.4747 -0.3549  6.1922
#>   2.7604  5.4356  1.1899  0.5558 -6.0146
#> 
#> (2,.,.) = 
#> -1.8066 -1.4108 -0.5840 -0.0159  1.7035
#>  -0.0152 -0.2216  0.4863  1.4318 -1.4032
#>  -0.1446 -0.6112 -1.1475 -0.8029  0.2463
#> 
#> (3,.,.) = 
#>  0.9536  3.3948  1.4783 -0.7987 -0.2265
#>   0.7731  1.8659  1.7980  0.5435 -0.4261
#>   1.4987 -0.1977 -0.8111 -0.2006 -1.7686
#> 
#> (4,.,.) = 
#> -0.5675 -2.0805 -1.8828 -1.1501  1.3443
#>   0.4234 -0.7309 -3.3073 -1.3807 -2.3517
#>  -0.4115  1.7024  1.7043 -0.6742  2.4133
#> 
#> (5,.,.) = 
#>  1.5411  2.5175  0.7976  0.5553 -3.0879
#>   0.0352  0.8683 -0.7220 -0.6061 -0.9098
#>   0.7159  0.2577  0.6310 -0.4343  0.9246
#> 
#> (6,.,.) = 
#>  1.5401  2.8134  2.4128  1.2909 -2.1292
#>  -1.5963 -3.0030 -2.2290 -1.1368  2.3657
#>  -1.6020 -1.1162  2.5632  1.8915  2.9051
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
