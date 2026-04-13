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
#> -2.2246 -2.4493 -3.3242 -3.1652 -1.1915
#>   2.4697  0.0101 -0.8409  1.0651  2.4446
#>   1.2064  0.7916  0.9494  1.2931  0.8235
#> 
#> (2,.,.) = 
#>  2.7297  2.1909  1.6202  2.4836  2.1860
#>  -0.3510 -1.2802 -1.4218 -0.9351 -0.0372
#>  -1.2877 -2.3920 -1.3626 -1.3978 -1.0487
#> 
#> (3,.,.) = 
#>  2.0976 -1.0032 -0.2303  1.3410  1.7004
#>   0.6718  2.6240  2.2460  1.6161  0.1577
#>  -0.3968 -1.7238 -2.6598 -1.7528  0.3546
#> 
#> (4,.,.) = 
#> -2.2009  0.1153 -0.1356 -1.2770 -2.0307
#>  -0.4446 -1.0156 -1.8170 -1.2777  0.0796
#>  -3.0499 -2.1578 -0.6490 -2.1473 -2.7638
#> 
#> (5,.,.) = 
#>  0.4080  1.5630  1.0458  0.6695  0.3054
#>   0.3024  1.2650  0.4895  0.4773  0.2068
#>   3.1655  1.3718  0.2853  1.9512  2.9679
#> 
#> (6,.,.) = 
#> -0.7776  0.4118  0.2037 -0.3372 -0.7428
#>  -3.3106 -0.4112  0.1246 -2.1356 -2.8726
#>  -0.1939 -0.6345  0.4683  0.1677 -0.3935
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
