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
#>  -1.3075  1.5814 -0.0483 -0.1708  0.7137
#>   5.5514 -4.8474 -1.9938  0.9806  0.0068
#>  -0.9548  1.8650 -0.4198  0.1665 -0.2547
#> 
#> (2,.,.) = 
#>  -0.5606  0.5911  0.6311 -0.2015 -0.2327
#>  -6.4861  0.2912  0.7419 -0.6089 -0.7799
#>  -1.8059  2.2605  2.7039 -0.9371 -0.3739
#> 
#> (3,.,.) = 
#>  -1.9561 -1.4880 -0.4929 -0.1241  0.5562
#>  -1.9570  0.2198 -1.8496  0.7492 -1.4496
#>  -4.1254  2.6275  2.5612 -1.1367  0.0599
#> 
#> (4,.,.) = 
#>   0.8789  0.7486 -3.4189  1.6821 -2.3399
#>  -1.9727  2.8829 -1.5546  0.7757 -1.7606
#>  -3.4294 -0.0575 -2.5564  0.8705 -1.4558
#> 
#> (5,.,.) = 
#>  -0.2840  1.8417 -2.3540  0.6611  0.9589
#>  -1.3936 -0.2534  1.9821 -0.8632  0.3836
#>  -0.6710 -0.0867 -1.6066  0.5853 -0.4957
#> 
#> (6,.,.) = 
#>  -4.0017  1.9895  1.1048 -0.7383  0.5726
#>   1.5025  1.5638  0.2941  0.0822 -0.1309
#>   3.1493 -0.8363 -0.1037  0.0397  1.2044
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
