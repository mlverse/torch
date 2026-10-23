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
#> -2.3264  0.0917  2.6300  0.6455  1.5309
#>   2.0732  0.1717 -1.9502  0.3575  0.4452
#>  -0.1495  0.1679  0.5570  1.3310  2.2260
#> 
#> (2,.,.) = 
#>  1.0518  0.3418  1.4944 -0.0223  3.8573
#>  -0.6270 -0.3958  1.0028 -0.2493  0.0555
#>  -1.4707 -0.2233 -1.1119  1.3182 -2.3282
#> 
#> (3,.,.) = 
#> -1.1783 -0.3143  0.4869 -0.5304 -1.6984
#>  -1.8621 -0.3753  1.6929 -1.4767 -1.9381
#>  -0.0002  1.5872  0.6507 -2.2582 -0.3381
#> 
#> (4,.,.) = 
#>  2.0108 -0.7460 -2.2875  3.1755  2.2932
#>   1.6052 -1.1837 -0.4182  0.3051  0.9825
#>  -0.4549 -0.8999 -1.0058 -0.7086 -3.5695
#> 
#> (5,.,.) = 
#> -0.4572  1.0531  1.0544 -4.0212 -2.7235
#>   1.4857  0.0898 -2.9968  1.3643 -0.8346
#>  -0.2757  2.5841  0.6928 -0.5323  2.0294
#> 
#> (6,.,.) = 
#> -2.3005 -0.0249  1.7593 -1.5999 -2.3008
#>  -0.5360  0.9336  0.5294  0.2021  0.9947
#>  -0.0927  0.1600  2.4216 -1.6863  1.6962
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
