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
#>   3.3188 -0.2444 -0.7501 -0.3815  0.8224
#>  -2.6192  0.4324 -0.1143  0.1220 -1.2974
#>   0.6051  1.1556  0.3450  0.9736 -0.4292
#> 
#> (2,.,.) = 
#>  -1.5832  1.2745  0.5855  1.1906 -1.7059
#>   0.9060  0.7955  0.8210  1.0378 -0.0707
#>   2.3014  2.1227  1.3206  2.3299 -0.7333
#> 
#> (3,.,.) = 
#>   0.4196  0.7349  1.5928  1.2690  0.7254
#>   1.4006  1.6513  1.2836  2.0062 -0.9801
#>  -1.9726 -0.2888 -0.9262 -0.5832 -1.8203
#> 
#> (4,.,.) = 
#>  -0.1560 -1.3260 -0.9787 -1.5951  1.4188
#>  -1.1353 -1.5197 -0.1392 -1.2998  1.4578
#>  -2.2463 -1.2474 -1.6556 -1.7837 -0.7116
#> 
#> (5,.,.) = 
#>   0.8931  0.3054  0.3849  0.4715  0.1263
#>  -1.1180  1.9154  0.7637  1.9221 -2.8939
#>   0.9251  0.8402  0.1820  0.8110 -0.7812
#> 
#> (6,.,.) = 
#>   2.1897 -2.1545  0.0262 -1.6782  4.0814
#>   3.2930 -0.8544  0.3177 -0.4048  2.6227
#>  -3.8502 -0.5857 -2.3360 -1.6818 -2.4839
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
