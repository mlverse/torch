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
#> -1.2316 -3.3557  1.7191 -1.9079  2.8922
#>  -0.5724 -1.6412  0.1821 -1.4801  1.6891
#>   0.4727  0.9590 -1.5638 -1.0863 -1.4280
#> 
#> (2,.,.) = 
#>  0.6009  2.2059  0.3122  2.3375 -2.9697
#>   0.2742  1.2243 -3.7903 -1.9268 -0.3650
#>   1.5978  1.7181  0.7600  0.2182 -3.5041
#> 
#> (3,.,.) = 
#>  0.4057 -0.4154  1.5662  0.6128  0.5831
#>   0.5586  1.3157 -1.0798  0.5474 -0.6323
#>   2.8814  2.8098 -0.9087 -1.1065 -3.9759
#> 
#> (4,.,.) = 
#> -0.6414 -0.1918 -1.0044 -0.5846  0.0985
#>   0.4514 -1.3601  3.2062 -0.0207  0.1013
#>   0.6954 -0.5662  2.6928  0.4043 -0.5421
#> 
#> (5,.,.) = 
#> -1.1659 -2.0613  0.9198 -0.3659  2.4857
#>   2.0135  3.5764 -3.4581 -0.7588 -3.4779
#>  -0.1807  0.5989 -1.7337 -0.1923  0.2021
#> 
#> (6,.,.) = 
#> -0.4909  0.5170 -3.1134 -1.0133  0.8332
#>  -0.3347  0.3123  0.2502  0.8995 -0.6823
#>  -0.5111  0.1498 -2.8655 -1.7326  0.4572
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
