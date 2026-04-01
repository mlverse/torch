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
#>  -1.3076 -0.7136  1.5694 -1.2255 -1.3021
#>   0.1862 -0.3293  0.2521  0.0667  0.3542
#>  -0.8719 -0.6306  0.8252 -0.6856 -0.3151
#> 
#> (2,.,.) = 
#>   0.1767 -0.0472 -0.6730  0.4307  0.8084
#>   1.1201  0.8486  0.4815  0.1372 -1.1888
#>  -2.9542 -3.0770 -1.3766 -0.0652  4.3959
#> 
#> (3,.,.) = 
#>   0.6854 -1.2043  0.6939  0.4050  1.2093
#>  -0.2183 -0.3090 -0.3481  0.1273  0.7388
#>  -0.6397 -0.9633  1.8909 -0.9546 -0.9047
#> 
#> (4,.,.) = 
#>  -1.3060 -1.0273  0.9466 -0.8598 -0.1022
#>  -0.6253  0.3331  1.3703 -1.0882 -2.1058
#>  -1.5293 -2.6798  1.0936 -0.5536  1.8424
#> 
#> (5,.,.) = 
#>  -0.5182 -0.2589 -0.1775 -0.1225  0.3101
#>   0.9416 -0.7878  0.5929  0.4543  0.9507
#>   0.9410  0.8598 -1.5870  1.0321  0.7016
#> 
#> (6,.,.) = 
#>   0.3379  0.7464 -1.9256  0.8864  1.0045
#>  -0.7117  0.6682 -0.3639 -0.4443 -0.6530
#>   1.3121  0.4231 -0.2138  0.6801  0.2802
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
