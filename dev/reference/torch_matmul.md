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
#>  -3.3383  1.7095  1.5505  1.3202  0.7373
#>   0.4999  0.8385 -0.9150  3.0240  2.5848
#>  -3.7388 -0.8051 -2.0847  1.0103  0.3094
#> 
#> (2,.,.) = 
#>  -2.1260 -0.7117 -0.9184 -1.2835 -1.5559
#>  -2.5329 -0.7982  0.7578  0.4486  0.6667
#>  -1.0867 -2.4199 -1.4825 -1.2009 -0.8595
#> 
#> (3,.,.) = 
#>  -0.5584  1.1284  0.2725 -0.4692 -0.8968
#>   2.0279 -1.7731 -0.1242 -1.5541 -0.7135
#>  -1.6542  0.9624 -1.3323  1.5785  0.6840
#> 
#> (4,.,.) = 
#>  -1.1366  2.2727 -1.1665  2.8530  1.6028
#>   1.3459  0.8765  3.6190 -1.3315 -0.5257
#>   2.3278 -0.7884 -1.5858  0.5188  0.6667
#> 
#> (5,.,.) = 
#>   1.9985 -2.0943  0.5737 -0.9939  0.1715
#>   0.7548  0.2744 -0.4659 -1.4867 -1.6641
#>   0.1369  0.6275 -1.0876 -0.8528 -1.3953
#> 
#> (6,.,.) = 
#>  -1.0519 -0.7666  0.3063  1.0695  1.3426
#>  -2.8600  1.1822  0.4347  0.9024  0.2557
#>  -1.6029  0.6836 -0.3447 -0.2579 -0.8098
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
