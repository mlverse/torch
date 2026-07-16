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
#> -2.7933  0.6624 -0.6565  1.0374 -0.0443
#>   1.5798  0.4898 -0.8328 -1.4364  0.7310
#>   0.0383  0.9695 -0.8743 -1.9584 -1.6266
#> 
#> (2,.,.) = 
#>  4.5536 -1.0259  0.4631 -0.9885  0.2367
#>   0.0789 -2.5664  4.1523  1.9165 -0.1124
#>   1.8055 -0.5063  0.8176 -1.2209 -1.8214
#> 
#> (3,.,.) = 
#> -0.9698  0.4218 -0.2972 -0.0055  0.8974
#>   3.6055  1.2859 -1.9015 -4.3201 -2.4502
#>  -1.5012 -0.7914  1.4392  1.5535  0.1015
#> 
#> (4,.,.) = 
#> -0.6281  0.8129 -1.3027 -0.1843 -0.1239
#>  -0.7290 -0.1889  0.3813  0.8130  1.6534
#>  -3.4920  0.8060 -1.9974  3.1918  1.4198
#> 
#> (5,.,.) = 
#> -0.3328  0.8543 -1.4315 -0.4691 -0.9944
#>  -2.4359 -0.9240  0.6125  3.9822  1.3563
#>   3.7558 -0.2171 -0.0416 -2.3681 -1.6401
#> 
#> (6,.,.) = 
#> -2.2988 -0.6447  0.8403  2.7583  1.6884
#>  -1.8895 -0.2360 -0.3908  2.7945  0.7145
#>  -0.1696  0.2585  0.0183 -0.5576  0.9286
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
