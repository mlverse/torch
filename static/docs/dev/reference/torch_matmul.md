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
#> -0.2847 -0.2056  0.0326  0.1454  0.3447
#>  -2.4444 -1.0586  1.3743 -2.1946 -2.7098
#>  -1.0868  0.1342 -2.7250  2.1309  1.2311
#> 
#> (2,.,.) = 
#>  1.3146  1.2300 -4.0652  1.1254  1.3690
#>   0.6763  0.4632  0.4639 -0.5020 -1.2063
#>  -0.9394 -0.5755  3.8237 -1.9571 -3.4698
#> 
#> (3,.,.) = 
#>  8.2923  3.7144 -3.7627  1.8550  4.0617
#>   1.3549  0.6137  0.1217 -1.0921 -0.8745
#>   1.4592  1.0068 -3.4580  2.3414  2.8467
#> 
#> (4,.,.) = 
#> -3.7483 -1.1841  0.9542  1.6885 -1.1678
#>   0.5525  0.3961 -2.9562 -0.5520  1.0593
#>   2.8941  1.7478 -5.3080  3.0002  4.2578
#> 
#> (5,.,.) = 
#> -0.4265  0.0495 -0.2534  0.1513 -0.5510
#>  -1.2407 -0.3920  0.3801 -1.3834 -1.8587
#>   1.3892  0.9239 -1.7161 -0.6946 -0.3682
#> 
#> (6,.,.) = 
#>  0.6700  0.8314 -2.1222  1.8513  0.9745
#>  -2.0124 -1.7000  3.5893 -0.7463 -0.2555
#>  -0.4916 -0.5607  0.7112 -1.0756 -0.1145
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
