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
#>   0.4983  0.7926  0.4711  0.3199  0.0124
#>   1.5114  3.1841  2.3441  0.5613  0.0131
#>   2.2426  0.8396  0.6727 -0.7624 -4.5649
#> 
#> (2,.,.) = 
#>  -1.3346  1.0979  0.5031 -0.0278  2.2336
#>  -1.3255 -0.7637  0.1094  0.4822  3.4256
#>  -4.2881  0.2803 -2.3445  2.1680  8.1279
#> 
#> (3,.,.) = 
#>  -0.9372 -0.8784 -1.0286  0.4853  1.5787
#>   1.1943  0.6005  0.1790 -0.3203 -2.5328
#>   2.2080 -0.1614  1.5039 -1.0410 -3.7084
#> 
#> (4,.,.) = 
#>  -2.7924 -1.5296 -1.0420  0.2058  4.2753
#>   0.1478  1.2970 -2.3331  1.3631 -0.8408
#>   2.5297 -1.1729  1.2762 -1.3181 -4.6017
#> 
#> (5,.,.) = 
#>  -2.8299 -1.4485 -0.6346  0.4822  5.2678
#>  -2.0220 -2.7096 -0.0300 -0.6538  3.0444
#>  -2.7331  1.4843  0.5263  0.7941  5.8642
#> 
#> (6,.,.) = 
#>  -2.9758 -2.1984 -1.2750  0.5464  5.1893
#>   2.6436  3.9242  2.1589 -0.2428 -3.7691
#>  -0.8673  1.3527 -1.1747  0.5598  0.5879
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
