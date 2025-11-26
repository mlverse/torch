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
#>   0.8490  0.8333  0.5233  1.3260  0.0601
#>  -1.1282 -0.5568  1.2358 -0.3763  0.9923
#>   1.3061  0.5610  1.5953  1.2128 -1.9591
#> 
#> (2,.,.) = 
#>   3.1260  0.7926  2.2851  1.9531 -5.2455
#>  -1.0893 -1.7984 -0.0651 -2.1402 -1.2645
#>  -0.6739 -1.5087 -1.4875 -1.4631  0.0910
#> 
#> (3,.,.) = 
#>   0.7308  0.9018  2.1053  1.7040 -0.1623
#>   1.5145 -0.4872  0.9408  0.2004 -3.7054
#>  -0.2198 -2.7246 -2.4014 -2.4302 -1.9127
#> 
#> (4,.,.) = 
#>  -0.0061  0.2580  0.4185  0.1600 -0.0322
#>   0.4531 -1.3670 -1.7679 -1.8300 -2.9533
#>   0.3819  1.1059  4.2803  2.2909  0.0672
#> 
#> (5,.,.) = 
#>   0.7829  2.3033  2.4649  2.5954  0.9476
#>  -1.6208  2.3400  2.3498  2.1103  5.5700
#>   1.5659  1.5563 -0.5045  1.4577 -0.6327
#> 
#> (6,.,.) = 
#>  -0.5109 -2.1546 -1.7250 -2.7524 -2.4295
#>  -2.9491 -3.0918 -2.7723 -4.0773  1.5182
#>  -0.3522  0.3803  5.5876  2.0297  0.3477
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
