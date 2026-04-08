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
#>  0.3113  0.0561  0.4099  2.6032  0.9277
#>  -0.4099 -1.7759 -0.6970 -0.0373  0.0802
#>  -0.6867 -1.1296  1.6128  0.7325  2.1699
#> 
#> (2,.,.) = 
#> -0.0014  0.1893 -0.3309 -2.3500 -1.0534
#>   1.4646  4.3367  1.6897 -1.2542 -0.9789
#>  -0.9540 -2.9519  0.4465  1.9121  2.1963
#> 
#> (3,.,.) = 
#>  0.3201  1.6068  0.7440 -1.6023 -0.4842
#>  -0.1756 -0.4960 -0.8521  3.2469  0.6302
#>   0.9834  2.7635  0.8053  1.2657 -0.1920
#> 
#> (4,.,.) = 
#> -0.1697 -1.4364 -0.2300 -0.2178  0.1409
#>  -0.9100 -0.6066 -1.4533 -1.9644 -0.9801
#>  -0.1666 -0.2293 -0.8870  0.2472 -0.4320
#> 
#> (5,.,.) = 
#>  1.3388  4.6333  2.9798 -4.3086 -0.9588
#>   1.0977  2.4823  0.7183 -3.0657 -1.7036
#>   0.0177 -1.3471  0.7785  0.8427  1.1033
#> 
#> (6,.,.) = 
#>  0.6296  0.8522  1.7586  0.2803  0.8191
#>   1.1835  3.3049  3.2387 -1.2233  0.6029
#>   0.8796  3.1157  2.1433  0.8486  0.7001
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
