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
#>  0.8679 -0.0758 -1.2006  0.2536 -1.5617
#>  -1.6077 -1.4283  2.3140  3.0286 -2.1373
#>  -2.4068 -0.2561  2.9835  1.2198  0.3264
#> 
#> (2,.,.) = 
#> -4.5892  0.7251  4.0428  1.7663 -0.9958
#>   3.0267 -0.1017 -0.7863 -1.5778 -1.1759
#>   0.0066  1.3159 -1.3659 -1.1930 -0.2565
#> 
#> (3,.,.) = 
#>  0.0562  0.5348 -3.8430  0.8865 -1.7517
#>   2.7043 -0.6718  2.5636 -0.7714 -3.6402
#>  -0.2839  0.1069 -5.2958  1.9299 -1.2881
#> 
#> (4,.,.) = 
#>  1.6263 -1.2472  4.5893  0.2711 -3.8470
#>   1.0596 -0.2546 -1.4905  0.0202 -0.3066
#>  -0.0339 -0.6997 -1.0635  0.7407  1.4397
#> 
#> (5,.,.) = 
#>  0.8454 -2.4268  2.4397  2.2731 -1.7711
#>   0.6991 -1.2878 -3.6071  1.9338  0.4188
#>   3.6404 -1.1336 -2.4608 -0.5512 -0.3576
#> 
#> (6,.,.) = 
#> -0.9667 -0.6476  0.5347  0.7020  2.2141
#>   0.2805 -3.1545  5.9242  3.8564 -5.6286
#>  -0.8102 -1.5617  3.0217  2.4733 -2.3747
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
