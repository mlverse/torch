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
#>  -0.1896  2.2923 -3.5556  0.2710 -5.2391
#>   1.4594 -2.6248 -0.3912 -1.2176  2.8288
#>   2.2026 -1.7681  1.4714 -0.8833 -3.0585
#> 
#> (2,.,.) = 
#>  -2.2112  0.6093  0.3145 -1.3723  2.0516
#>   0.0696  0.5361 -1.1220  0.7856 -0.0639
#>  -1.4103  1.8371  1.9760  0.8671 -1.6652
#> 
#> (3,.,.) = 
#>  -2.1054  2.9605 -0.8696  1.5514 -0.6635
#>   0.6586 -0.5713 -0.2573 -0.6293 -1.0588
#>   0.8432 -0.2970 -1.5968  0.0168 -0.6782
#> 
#> (4,.,.) = 
#>  -1.7249  1.9185 -1.0045  0.6169  0.2302
#>  -1.4068  0.0743  0.0495 -1.0743  2.1494
#>  -0.2282 -0.8722  0.3641 -0.7662  2.2499
#> 
#> (5,.,.) = 
#>   3.1026 -3.1537 -3.2494 -0.7008  2.0973
#>   0.0023  1.7401  0.3037  1.5164 -3.6464
#>   1.7949 -1.2389 -1.8723 -0.6155 -1.3166
#> 
#> (6,.,.) = 
#>   0.1797  1.6513  1.1065  2.2846 -2.8872
#>   2.6373 -2.3616 -3.8029 -0.9789  0.4803
#>  -0.4356  0.1742 -0.8180  0.1507  1.4918
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
