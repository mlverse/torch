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
#>  -1.2555 -1.0113 -2.0136 -4.5048  0.1466
#>  -0.9243 -4.2404 -1.3942 -5.1508 -0.8188
#>  -1.1073 -0.4880 -0.9573 -2.2759  0.2331
#> 
#> (2,.,.) = 
#>  -2.1098 -1.4941  0.7192 -0.0617  0.2685
#>   0.7231  0.0353  0.0264  0.6640 -0.3960
#>  -1.0313 -1.7450 -0.6310 -2.9570  0.1102
#> 
#> (3,.,.) = 
#>  -1.9813  0.0597 -1.9704 -3.4382  0.4312
#>  -4.6756 -2.7479  0.6620 -1.7979  0.8775
#>   0.1537  0.3761  0.4672  1.4217 -0.0893
#> 
#> (4,.,.) = 
#>  -1.7383 -0.7491 -1.1638 -2.9126  0.3513
#>   4.0591 -2.4185 -0.3979 -1.6308 -1.7157
#>   1.9668  3.7129  0.7659  4.3394  0.1987
#> 
#> (5,.,.) = 
#>   1.2555  0.9192  0.1970  0.4762  0.1353
#>   0.0206 -0.8369 -1.2329 -2.4215 -0.3313
#>   0.0422  1.8939  0.9061  2.7588  0.4660
#> 
#> (6,.,.) = 
#>  -1.4437 -1.9943 -1.0384 -3.1962 -0.1270
#>   0.9456  2.2802 -0.1716  2.0572 -0.0435
#>  -0.2845  2.1990 -0.1112  1.0321  0.6542
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
