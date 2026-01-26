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
#>  -0.4515 -2.7717 -2.1057  0.8147  1.5990
#>  -0.6730  1.6797 -0.0939 -0.3142  0.4818
#>   2.0810 -1.0308 -1.3592 -1.6602 -1.1273
#> 
#> (2,.,.) = 
#>  -2.5061  0.9402  0.4685  1.0487  1.7300
#>  -1.5958  2.7911  4.4555  0.7750 -1.1218
#>  -1.9740  1.8539  2.5871  2.4278  0.6214
#> 
#> (3,.,.) = 
#>   0.6327  0.0824  1.5141  0.2874 -1.1868
#>   1.5875  1.0057 -0.5157 -2.9020 -1.5471
#>  -2.4973  2.1820  6.3166  0.5907 -1.6644
#> 
#> (4,.,.) = 
#>  -1.9044 -2.4333 -1.9520  3.1067  3.1474
#>  -0.5610  2.1977  1.5372  2.1880  0.2329
#>  -0.2688 -2.1147  1.0422  2.5049  0.2156
#> 
#> (5,.,.) = 
#>   1.9525 -0.7332 -1.1495 -2.5359 -1.4016
#>  -1.5910 -0.2419 -1.4687  2.3593  2.5511
#>   2.4432 -0.2409  0.1338 -3.1193 -2.5979
#> 
#> (6,.,.) = 
#>   3.2440 -1.2845  0.0028 -3.3491 -3.1648
#>   1.7147 -0.8636 -1.8302 -1.5687 -0.5853
#>  -1.6577  0.0149  3.0710  2.9053  0.1934
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
