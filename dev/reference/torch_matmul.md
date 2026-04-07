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
#> -0.7393  2.6070 -0.8318  1.0092 -2.0181
#>   0.7662  0.3322  1.9797  0.4861  2.9693
#>  -1.2975 -2.7562 -2.1738 -0.0779 -0.9389
#> 
#> (2,.,.) = 
#>  0.9396  1.8566  0.2796 -2.5184 -1.9551
#>   2.5368  3.1129  2.8371 -0.9163 -0.3546
#>  -1.8975 -0.6983 -1.5649  2.1335  0.4551
#> 
#> (3,.,.) = 
#> -2.5635 -0.9962 -2.6415  1.4876 -0.3676
#>   3.1186  2.4049  2.7010 -2.7722 -1.2894
#>  -1.7970  1.9373 -1.4783  0.2437 -0.4932
#> 
#> (4,.,.) = 
#> -0.4946 -0.2995 -0.9003  0.1557 -1.0041
#>   0.8758 -0.5250  1.3603  0.3556  1.5669
#>  -2.7955 -1.4608 -1.7601 -0.0765  3.2042
#> 
#> (5,.,.) = 
#>  0.8604  0.5711 -0.0708 -1.8739 -2.1521
#>  -1.6241  0.2300 -1.7763  0.3564 -0.8133
#>   3.6221  0.6099  2.4434 -3.3225 -2.1069
#> 
#> (6,.,.) = 
#> -2.1049  0.7025 -1.6558  1.3047  0.1986
#>   2.4547 -0.4350  3.3510 -2.1242  3.4542
#>  -0.0899 -0.7591 -0.9433 -0.6412 -1.6690
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
