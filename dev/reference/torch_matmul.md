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
#> -2.6668 -3.0381  3.4918 -3.0393  1.3457
#>   1.3005  3.2549 -0.7097 -2.8757 -3.2269
#>  -0.6670  1.7960 -1.8552  0.1504 -4.2043
#> 
#> (2,.,.) = 
#>  1.6140  0.5100 -0.2333 -0.2390  4.7231
#>   1.2081 -1.7147  3.1862 -0.7596 -0.5655
#>  -2.7640 -0.2810 -1.3272  1.4309 -5.2297
#> 
#> (3,.,.) = 
#>  0.3424  0.3566  2.4532 -4.1342 -1.2255
#>   0.2898 -0.8272  0.9699 -0.2950  1.9956
#>  -2.2297 -3.7433 -0.4335  6.2821 -2.3875
#> 
#> (4,.,.) = 
#> -0.7949 -0.8059  1.0981 -1.6888  2.3024
#>  -2.2486 -0.1359  0.0686 -1.3516 -2.6692
#>  -0.2331  3.5222 -2.5512 -3.0222  1.4731
#> 
#> (5,.,.) = 
#>  0.8863 -0.5672  2.3681 -1.7766 -0.9616
#>   0.7699 -0.9658  3.7495 -4.9790  4.8847
#>  -2.8361  0.5743 -2.6576  2.6945 -8.0089
#> 
#> (6,.,.) = 
#>  0.4856 -1.5231  1.0543  1.9553 -1.3669
#>   0.7087  0.3503  2.5760 -5.0831  3.0245
#>  -0.7838  0.7872 -0.4480 -1.5507  0.1638
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
