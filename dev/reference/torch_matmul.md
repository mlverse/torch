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
#>  5.0581  5.3686  3.3121  4.9243  0.1434
#>   3.4109  2.0163  2.7843  3.4594 -1.0439
#>  -5.0154  1.7128 -2.9354 -3.9893  0.3785
#> 
#> (2,.,.) = 
#> -3.2858  0.6035 -2.9890 -4.0946  1.5721
#>  -2.7505 -0.2696 -2.2150 -2.4842  1.0606
#>  -0.6800 -1.2534 -1.3727 -2.3124  0.9855
#> 
#> (3,.,.) = 
#> -1.0192  0.9793 -1.1486 -1.6148  0.8531
#>  -1.7103 -2.4848 -1.8965 -2.7652  0.8741
#>  -1.4240  1.4442  0.7873  0.3051 -2.1963
#> 
#> (4,.,.) = 
#> -1.4105 -1.9074 -1.8366 -2.0798  1.3122
#>   0.3798 -2.7586  0.8680  1.2388 -1.1578
#>   2.1822  0.7287  1.6577  2.1636 -0.5260
#> 
#> (5,.,.) = 
#>  1.8032 -0.7888  0.3761  0.8360  0.8578
#>   0.8091  0.0710  1.3339  0.8867 -1.5010
#>   1.7069 -4.3477  0.9146  1.1122 -0.5057
#> 
#> (6,.,.) = 
#> -2.3035  4.0654 -1.3802 -1.5718  0.7127
#>   1.0630 -0.0445  0.9901  0.3860 -0.8911
#>  -2.7804 -0.2025 -1.8222 -3.0811  0.1009
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
