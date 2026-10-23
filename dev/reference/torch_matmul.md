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
#>  2.1264  2.7426 -1.2276 -1.8647 -2.3867
#>   1.1365 -0.3185  1.6266 -0.8828 -0.7199
#>   1.7243  1.6593  0.6549 -1.7651 -1.5473
#> 
#> (2,.,.) = 
#>  2.1314 -1.2056 -4.2201 -1.7295  0.7247
#>  -2.4844 -2.9763 -1.0582  3.1699  1.7584
#>   0.7655  0.2705 -0.4507 -0.8891 -0.0625
#> 
#> (3,.,.) = 
#> -1.9362 -3.6488  4.2660  3.0482  0.3563
#>  -1.3359 -0.2577  4.5856  2.0481 -1.5565
#>  -0.1204  1.4188 -2.5093  0.2841 -0.6307
#> 
#> (4,.,.) = 
#> -0.3769  1.1262  2.5355 -0.1152 -0.5302
#>  -0.5020 -0.6901  3.5216  1.1295 -1.0812
#>   1.7043  1.2137 -3.3764 -2.2942  0.4033
#> 
#> (5,.,.) = 
#>  0.7228 -0.4449 -1.4279 -0.8157  0.5955
#>   0.6883 -1.9427  2.7909  0.0369 -0.4527
#>   1.8685  1.9845 -4.0008 -2.0398 -0.6122
#> 
#> (6,.,.) = 
#> -0.5831 -2.1141  4.9694  1.6451 -1.0302
#>   1.9213  2.1857  0.0625 -2.2267 -1.4391
#>  -3.6301 -3.8046  2.7441  4.3395  1.6973
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
