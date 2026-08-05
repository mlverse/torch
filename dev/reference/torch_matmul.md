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
#>   3.7811   3.3426  -3.5085  -4.0467   3.6536
#>    1.4796  -1.0116   5.3021   1.5103   4.9664
#>    8.2683   0.6841   1.5710  -1.9863  10.4645
#> 
#> (2,.,.) = 
#> -1.9780 -0.5999  2.9860 -2.2039  0.9080
#>  -1.2713 -0.4926  0.3666  2.6266 -1.9863
#>   0.8610 -1.6008  4.2686  2.6038  2.8727
#> 
#> (3,.,.) = 
#>  0.7111  1.1811 -0.5718 -2.4893  1.5480
#>   3.6065  1.0598 -4.6645  4.0169 -1.0188
#>   3.2553 -0.3625  2.7239  0.1232  5.3048
#> 
#> (4,.,.) = 
#>  1.0880 -0.2248  2.1343 -0.1253  2.7977
#>  -0.7319 -0.5617  3.3588  1.3808  1.3968
#>   0.4331  0.9455 -1.2927  4.7235 -1.6197
#> 
#> (5,.,.) = 
#>  1.5916 -2.2206  2.6338  3.3550  1.7055
#>  -0.4679  1.8457 -0.0031 -5.1815  1.9838
#>   0.5451  0.7130 -1.6465 -0.4363 -0.3610
#> 
#> (6,.,.) = 
#> -2.1450 -0.6758 -0.4174  2.0507 -3.4260
#>   1.7788 -0.3052 -1.9033 -0.9455  0.3134
#>   1.3253  0.6388  0.8554 -0.1890  2.3986
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
