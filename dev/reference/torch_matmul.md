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
#>  0.7484  0.2283 -1.6442 -1.2946  1.1174
#>   2.6577  2.8232 -0.9193  2.7386  3.7280
#>  -0.7084  0.4992  0.8536  0.9753 -0.2916
#> 
#> (2,.,.) = 
#> -3.2713 -3.6121  1.5084 -0.7029 -3.9615
#>  -3.7030  1.0965 -1.3227  4.7404  2.1667
#>  -3.2402 -0.5993 -0.1254  2.0541 -0.3550
#> 
#> (3,.,.) = 
#> -2.5087  2.5509 -1.0363  6.1225  3.4365
#>   2.0312 -3.6040  2.0311 -3.4707 -3.8344
#>   3.9086  0.3223  1.4355 -2.8389 -0.8751
#> 
#> (4,.,.) = 
#> -0.3068  0.0905 -0.1692  0.7954  0.3734
#>  -5.9411 -0.1735 -2.5560  5.2070  2.0860
#>   3.4889 -0.5783  2.1024 -4.9183 -2.6719
#> 
#> (5,.,.) = 
#> -0.8915  2.1171 -1.5280  3.2934  3.0605
#>  -4.5453 -1.7567 -0.4980  3.3172 -0.4895
#>   2.6590 -2.2973  1.0212 -3.6826 -2.3744
#> 
#> (6,.,.) = 
#>  3.0391 -0.3688 -0.4764 -4.1018 -0.4124
#>  -0.3924 -2.0213  1.4150 -1.2055 -2.5386
#>   3.4961 -1.4409  1.5651 -5.3723 -2.7975
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
