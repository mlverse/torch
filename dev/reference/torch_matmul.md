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
#> -2.0225 -1.3409  1.8039  0.8969  2.2103
#>  -1.5654 -1.0363  0.8699 -0.8953  1.1257
#>  -2.2185  0.5298  4.6800  4.4986  4.6394
#> 
#> (2,.,.) = 
#>  0.8380 -0.3492 -1.6335 -2.3447 -2.2351
#>  -1.1281 -1.2957 -0.5096 -1.4213  0.4472
#>  -1.3465 -0.5960 -0.3862  0.9152  2.4563
#> 
#> (3,.,.) = 
#>  1.8268  1.5830 -0.5505  0.9080 -1.2773
#>   0.9654 -0.6850 -4.8257 -4.9753 -2.7921
#>  -0.0823 -0.4362 -0.5211  0.1954  0.1080
#> 
#> (4,.,.) = 
#> -0.6414 -1.9206 -3.8209  0.3132  1.2633
#>   0.2097 -0.2900 -0.4881  1.1716  0.1887
#>  -0.2099 -0.4475  0.5014  1.3267  0.4885
#> 
#> (5,.,.) = 
#> -1.2828 -1.2997 -1.5889  2.2987  2.9608
#>  -0.8178 -1.0096 -0.8464 -1.0719  0.4324
#>  -1.4826 -1.2921 -0.6965  0.5474  2.1020
#> 
#> (6,.,.) = 
#> -0.6670  0.8768  4.4973  3.7229  1.9308
#>   0.7372 -0.0291 -2.5902 -0.6901 -0.6389
#>  -2.3143 -0.2659  5.6225  1.6744  2.4489
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
