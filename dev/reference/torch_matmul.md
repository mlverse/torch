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
#> -4.1551 -0.9688  4.3982  2.3996  2.6467
#>  -3.5380 -0.6614  2.4932  0.7056 -1.5668
#>   0.9671 -0.1909 -1.0370 -0.3367 -0.1169
#> 
#> (2,.,.) = 
#>  1.0463  1.2342  0.3244 -1.7380 -2.1719
#>   1.1065  3.0614  0.6690 -0.6734  0.3668
#>  -5.3616 -3.1618  3.1160  2.9659  1.4157
#> 
#> (3,.,.) = 
#>  2.1417 -1.1433 -3.5644 -0.4587 -0.4974
#>  -2.4076 -3.4672  0.4011  1.3941  0.1816
#>  -1.8377 -2.3802  1.5997  1.2422  1.4362
#> 
#> (4,.,.) = 
#> -4.1544 -0.9621  1.5297  2.5201  0.8567
#>   2.4282  2.0210 -1.1702 -0.5707  1.1786
#>   3.0148  1.0374 -3.7490 -1.6400 -2.1197
#> 
#> (5,.,.) = 
#> -1.2733 -1.0846  0.1360  0.1604 -1.2610
#>  -0.0943  0.1632  0.5033  0.7872  1.9342
#>   5.0464  0.7723 -4.0160 -1.5276  0.7804
#> 
#> (6,.,.) = 
#>  0.6075 -1.7505  0.6308  0.8087  3.1664
#>  -1.9718  1.6403  2.9270  0.2564 -0.1931
#>   1.2008 -0.5769  0.0221 -0.7860  0.0113
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
