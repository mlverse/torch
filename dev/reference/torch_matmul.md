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
#> -1.0566  0.7563  2.0492  1.0656 -0.2085
#>   2.7902  0.2119  0.4422 -1.5567 -0.3309
#>  -8.5569  1.0731 -0.7109  5.5543  0.2020
#> 
#> (2,.,.) = 
#>  1.9022 -2.5723 -2.6908 -2.5379  0.4267
#>   5.1281 -0.2453 -1.3088 -2.6089  2.1040
#>  -2.1408  1.8961  0.9531  2.4687  0.5260
#> 
#> (3,.,.) = 
#>  1.9011 -1.9717 -1.1649 -2.2673 -0.0810
#>  -9.1941  0.0197 -3.1083  5.5354  1.3856
#>   3.8468 -2.1976 -0.4638 -3.6306 -0.6465
#> 
#> (4,.,.) = 
#> -3.8973 -3.4971 -2.8737  0.3395  0.6490
#>  -1.2173 -3.4583 -2.8932 -1.2745  0.3012
#>   1.0053 -0.4714  0.1639 -0.9736 -0.5363
#> 
#> (5,.,.) = 
#>  3.8037 -0.7282 -0.2474 -2.7569 -0.5702
#>  -6.7361 -0.0825 -0.1030  3.7528 -0.3147
#>  -2.4680  0.0329  0.9088  1.2258 -0.9901
#> 
#> (6,.,.) = 
#>  2.2516 -0.5034  0.3158 -1.6695 -0.4036
#>   7.0077  4.2576  5.6339 -1.8712 -1.7417
#>   1.5025  0.4157  0.3620 -0.7186 -0.3875
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
