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
#>  -3.4145  1.7370 -3.5201  2.0091 -2.6248
#>   0.6349 -1.4562 -2.2569  0.6543  1.3262
#>   1.1895  1.7645  2.4652  4.2799 -1.0080
#> 
#> (2,.,.) = 
#>  -1.3107  1.4300 -0.0312  0.6870 -1.2632
#>   3.4538 -0.9599  1.5830  2.1379  2.9411
#>  -0.8100  0.4511 -1.4199  0.9645 -0.1955
#> 
#> (3,.,.) = 
#>  -0.1405 -0.9124 -2.3647  0.3050  0.8460
#>  -0.1232 -0.2973 -0.6014 -1.1476  1.0640
#>   2.2939  0.7600  2.0578  3.4384  1.3380
#> 
#> (4,.,.) = 
#>   2.6599  0.2556  4.2153 -0.3927  2.0302
#>   0.8831 -0.5773  1.4500  0.3101 -1.1696
#>   2.3044  1.3984  2.6760  4.2451  0.9740
#> 
#> (5,.,.) = 
#>   0.9634 -3.9481 -3.7939 -2.2098  2.2080
#>   3.4321 -0.6282  5.7620 -2.0635  1.7718
#>   0.3562  0.1057  1.4063 -2.0540  1.2025
#> 
#> (6,.,.) = 
#>  -1.5008  0.3630 -0.7645 -1.9960 -0.2201
#>  -1.2469  2.3566  2.3370  1.4246 -3.3245
#>  -4.0938  0.7902 -0.9616 -3.8363 -4.1972
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
