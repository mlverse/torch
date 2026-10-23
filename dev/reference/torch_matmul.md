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
#>  0.6914  0.7204  1.5522 -0.4049  0.0259
#>   1.6050  1.8012  2.0502 -1.7055 -0.9048
#>   0.5042  0.7988  1.0060 -0.8753 -0.1989
#> 
#> (2,.,.) = 
#> -0.6574  0.4629  1.3200 -1.0130  0.7796
#>   0.4623  1.3593  2.6182 -1.4337  0.6180
#>   0.1019  0.6432  1.7142 -0.2414  1.9056
#> 
#> (3,.,.) = 
#>  2.0984  2.1459  3.0746 -1.3193  0.4246
#>   0.0332  0.1599  0.0870 -0.2839 -0.1328
#>  -2.0916 -1.9711 -3.7599  0.9569 -0.2204
#> 
#> (4,.,.) = 
#>  0.8037  1.1071  0.9057 -1.2044 -0.3099
#>   1.5514  1.3649  1.3847 -1.5180 -2.4420
#>   0.9017  0.8686  1.6095 -0.5772 -0.3473
#> 
#> (5,.,.) = 
#>  2.4519  3.1244  4.0932 -2.6427  0.2823
#>  -0.4157 -0.3825 -0.2610  0.7646  1.7190
#>  -1.0641 -0.2538  1.1279  0.2400  2.0244
#> 
#> (6,.,.) = 
#>  0.2453  1.2603  2.8883 -0.8201  2.6555
#>  -0.7803 -1.8595 -2.2221  2.3562  0.2309
#>  -1.9787 -1.7405 -1.6470  1.6380  2.0250
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
