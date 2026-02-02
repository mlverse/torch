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
#>  -0.8102 -0.7225  2.0502  0.3883 -1.5171
#>   0.2892 -0.8387 -1.6410 -0.5570 -0.5418
#>   1.6378 -0.2350  1.1532  1.1421  0.9440
#> 
#> (2,.,.) = 
#>   1.0211  0.2427  2.1276 -0.2030  2.1924
#>  -1.0417  0.8691 -0.3650 -0.4649  0.0606
#>  -2.1131 -0.1485  5.2792 -0.8278 -0.4625
#> 
#> (3,.,.) = 
#>  -2.9924 -0.0170  5.3234 -1.6210 -0.6283
#>   1.0621 -1.4906  0.5001  0.9439 -1.1016
#>  -3.5497  0.8427  5.5555 -2.6519  0.6349
#> 
#> (4,.,.) = 
#>  -1.8599 -0.1352  0.0404 -1.3476 -1.1073
#>   1.6368 -0.2436 -3.0056 -0.1030  0.9043
#>   0.4538  0.3749 -0.9820 -1.1431  1.6694
#> 
#> (5,.,.) = 
#>  -0.2936 -0.1512  0.3172 -1.6021  0.9782
#>   2.3795 -0.1150  1.2611  1.3267  1.7993
#>   2.3088  0.6852 -1.8162  1.0921  1.9759
#> 
#> (6,.,.) = 
#>  -3.0920  1.7867  2.0649 -0.9896 -0.1871
#>  -0.2191  0.8061  4.0754  0.6530  1.1561
#>  -0.2699  1.8064 -1.0592 -1.1351  2.3790
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
