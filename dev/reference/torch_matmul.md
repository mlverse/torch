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
#>  1.2390  0.4126  1.0041 -2.4203  0.0889
#>   0.3426 -0.8232  2.9705 -2.6275  1.3521
#>   0.1174 -1.0266  2.0926  0.0165  0.7418
#> 
#> (2,.,.) = 
#>  0.5216  0.2126  1.0939 -2.9272  0.0470
#>   1.2911  0.0905 -1.1398  4.3467 -1.0351
#>   0.5427 -1.1318  4.5347 -3.8912  4.0981
#> 
#> (3,.,.) = 
#> -1.9343 -0.9747 -0.6933  4.0079  2.4763
#>   0.9078  0.2881 -0.8373  1.7155 -1.7703
#>  -0.0020 -2.0066  4.1783  0.7541  5.4243
#> 
#> (4,.,.) = 
#>  1.3546 -0.1114  2.2787 -3.1170 -0.3288
#>  -0.7912 -0.1935 -0.3185  1.5972  3.5456
#>   0.4659 -0.3997  0.6127  1.0696 -0.2230
#> 
#> (5,.,.) = 
#>  1.5766  0.8869 -1.5233  0.6957 -4.8525
#>   0.1689 -0.5782 -0.1294  4.0374  1.8736
#>   0.4436  0.3026 -0.2420 -0.8111 -2.1705
#> 
#> (6,.,.) = 
#> -0.1746  0.3174 -2.6510  4.7511 -0.6917
#>  -1.0386 -1.5925  4.1221 -2.3609  6.9141
#>   0.6110  0.3800  1.0282 -4.0196 -1.7521
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
