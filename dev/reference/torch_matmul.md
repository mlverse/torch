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
#>   1.4813  1.1747  2.1258  0.0523  1.6960
#>  -0.6752  0.8127  2.1262  2.4007 -2.3497
#>  -0.0319  0.0582 -1.9989 -1.1229  0.4143
#> 
#> (2,.,.) = 
#>  -0.6011  1.0226  4.9417  3.9023 -2.9352
#>   0.7764  0.3424  1.9522  0.1394  1.0028
#>  -0.4359 -0.9560 -1.3498 -1.0660  0.2673
#> 
#> (3,.,.) = 
#>   2.3540  1.2489  4.0009 -0.0298  3.0307
#>   0.1249 -0.2875  3.9860  4.4085 -2.0800
#>  -0.7391 -0.4607 -0.0228  2.1430 -2.1391
#> 
#> (4,.,.) = 
#>   0.5126  0.1769  3.8859  3.5024 -1.1781
#>  -1.0146 -0.0226 -1.4875  0.0211 -1.4395
#>  -2.0196 -0.6508  1.1040  3.6269 -4.4686
#> 
#> (5,.,.) = 
#>   0.6965 -0.6110 -3.7974 -2.9902  2.4864
#>   0.1907 -0.5672  0.3461  0.8487 -0.0905
#>   0.6512 -0.5167 -1.1665 -1.5380  1.7977
#> 
#> (6,.,.) = 
#>   3.1163  1.3853  2.6111 -0.7880  4.2431
#>  -0.9707 -0.4988 -3.2439 -1.1807 -0.6892
#>  -1.4465 -0.2341 -0.3425  1.2114 -2.5265
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
