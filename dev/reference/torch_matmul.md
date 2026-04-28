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
#>  1.3977 -0.8071 -1.3793  6.6738 -3.3532
#>   2.1333  2.1260  3.8244 -0.6445  3.7818
#>  -0.5757  1.3252  1.1835 -2.8181 -3.2426
#> 
#> (2,.,.) = 
#>  0.9607 -0.8200 -0.9338  4.3632  0.8131
#>  -0.4915 -0.4180 -1.5332  4.8285  2.5968
#>   0.3201  0.0256 -0.5937  3.4723 -1.8644
#> 
#> (3,.,.) = 
#>  0.3961 -0.9377 -1.7421  4.5634 -3.1279
#>   1.0760 -1.4209 -1.4793  4.3089 -0.3532
#>   1.7756 -0.4999  0.5755  1.2381  1.0220
#> 
#> (4,.,.) = 
#>  0.0567  2.7418  2.3177  0.3835 -2.6114
#>   0.9727 -0.5539  0.0742  1.8428  4.2935
#>  -1.8755  1.0241  0.0448 -0.2665  4.8501
#> 
#> (5,.,.) = 
#> -3.1776 -1.8429 -2.1443 -9.0234 -1.4406
#>  -1.9297 -0.1384 -1.2468 -1.0086  0.0282
#>   2.8737 -2.1020 -1.1590  5.4398 -1.1498
#> 
#> (6,.,.) = 
#> -1.1527  0.9059  1.2084 -4.9728  2.1403
#>  -0.7745 -0.9816 -1.4871  0.0142 -0.6364
#>  -2.2127  0.1432 -1.3725  0.1577  0.7092
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
