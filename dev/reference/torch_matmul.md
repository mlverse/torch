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
#>   0.6993  0.3017 -0.2429 -0.7508 -0.4182
#>  -1.4298  0.6945  1.0976 -2.2143  1.0502
#>   2.1376 -1.0362 -1.5713 -1.3525  2.9948
#> 
#> (2,.,.) = 
#>   2.2354  1.4386  0.0922 -0.1385 -0.5424
#>  -0.3981  3.6325  2.6351 -0.4321 -3.5548
#>  -2.9113 -0.7774  0.5014  4.6942 -5.2223
#> 
#> (3,.,.) = 
#>   4.3429  2.0500 -0.0368  0.1426  1.1270
#>  -1.0431 -0.7294 -0.1718  0.8367 -0.9143
#>  -0.5324 -0.7287 -0.3065  1.4255 -0.8376
#> 
#> (4,.,.) = 
#>  -1.3188 -0.8295 -0.1451  1.7400 -1.8070
#>   2.5858  1.0492 -0.1017 -4.6540  5.2966
#>  -3.0760  0.0439  1.1554  2.6757 -4.1702
#> 
#> (5,.,.) = 
#>  -0.7304 -0.2129  0.0678  0.6545 -1.1037
#>   1.9522  1.7215  0.4495  2.7277 -3.1829
#>  -1.9596 -1.1609 -0.1571  2.6702 -2.7766
#> 
#> (6,.,.) = 
#>  -3.2918  0.2854  1.4001  1.3659 -3.3708
#>  -0.9278  1.9610  2.0049  1.4886 -1.9079
#>  -0.3235 -0.7284 -0.4514  0.0667  0.0743
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
