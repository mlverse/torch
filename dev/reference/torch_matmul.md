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
#> -5.8698 -5.4001  1.3784  0.2533  5.5591
#>   1.7434 -0.9659 -0.4977  0.4521  0.5951
#>   3.5775  0.5674 -0.4012 -0.1876  1.0479
#> 
#> (2,.,.) = 
#> -0.7888 -0.3507 -0.4791 -1.4431 -1.1610
#>   0.6050  0.9123  0.4616 -0.8665  1.4517
#>  -2.1396  0.6897  1.4534 -0.2899  2.4839
#> 
#> (3,.,.) = 
#> -1.1617  0.4261  1.2467  1.5481  1.9958
#>  -2.8781 -2.7298  0.2327 -2.3591  2.5330
#>   1.0882  1.3831  1.5326 -0.9107  4.9163
#> 
#> (4,.,.) = 
#> -4.1583 -2.7306 -0.0533 -0.7407 -0.2352
#>   1.5253  4.0159  1.1682  0.8589  0.3488
#>  -1.3049  1.1944  0.9968 -1.1991  1.5319
#> 
#> (5,.,.) = 
#> -1.7234 -2.7446 -1.7478  0.7128 -4.5741
#>  -3.4807  1.3030  1.8674  1.3192  1.3313
#>   1.9962 -0.2541 -0.2949 -0.0792  0.9567
#> 
#> (6,.,.) = 
#>  1.8035  1.4472 -1.5863  1.0303 -5.8771
#>  -2.4271 -2.0830 -0.5525 -0.9639 -1.0847
#>   0.9957  1.9259  0.9611  0.0703  1.8794
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
