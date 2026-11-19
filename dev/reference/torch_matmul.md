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
#> -2.1131 -2.8321  0.8746  1.4318 -0.7625
#>  -3.0871 -0.0407  1.3253  2.1502 -2.2437
#>   1.7694  4.0566 -1.5410 -1.6026  0.4228
#> 
#> (2,.,.) = 
#>  0.0601 -2.2218 -0.7847 -0.3502  1.1417
#>   0.4787  0.4358  0.6500 -0.0592 -0.4703
#>   1.3543 -1.0811  0.2546 -0.3901  1.3711
#> 
#> (3,.,.) = 
#> -0.5792  0.0421 -1.3524 -0.4571 -0.0420
#>  -1.1380 -4.8449  0.8666  1.3863  1.5126
#>   2.5167 -4.3100  0.6755 -1.0430  2.0662
#> 
#> (4,.,.) = 
#> -4.2840  2.6737 -1.2434  1.4473 -2.8456
#>  -2.6743  0.6448  2.2986  2.6463 -1.9975
#>   0.0698 -0.8716 -0.5085 -0.6295 -0.4155
#> 
#> (5,.,.) = 
#> -0.0337  4.8076 -1.4021 -0.9089 -1.5327
#>   0.8487  1.9295 -0.8168 -0.9714 -0.1915
#>   2.6819  0.7698 -2.4220 -2.9698  0.9584
#> 
#> (6,.,.) = 
#>  3.3547 -3.7391  0.8500 -1.0240  3.1987
#>  -0.3578 -0.5093  0.0948  0.2260 -0.0755
#>  -0.0290 -2.9951  1.4018  0.8728  0.7917
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
