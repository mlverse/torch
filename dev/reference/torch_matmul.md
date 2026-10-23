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
#>  0.3856  0.3086 -1.1220 -1.0800  0.2467
#>  -2.6108 -0.1826 -0.9453  0.9506 -0.7741
#>  -0.6145  0.7585 -0.3329  0.4005  0.5677
#> 
#> (2,.,.) = 
#> -1.3673 -0.0956 -0.7396  0.5135 -0.2238
#>  -1.3079  2.2062 -2.9713 -1.3099  1.0410
#>   0.6608 -1.4875  0.4116 -0.2452 -0.8913
#> 
#> (3,.,.) = 
#>  0.4520  0.9525 -0.4406 -0.7645  0.5479
#>   1.5965 -0.9781  1.6162  0.7454  0.3571
#>  -1.6528  1.3359 -0.5465  0.0840 -0.1101
#> 
#> (4,.,.) = 
#>  1.0748 -0.8231  1.3994  0.6416  0.0592
#>   0.3359 -0.2556  0.4147 -0.1055 -0.2596
#>  -3.0531  1.9931 -1.9846 -0.2805 -0.2572
#> 
#> (5,.,.) = 
#>  2.2685 -2.4443  2.3855  0.9482 -0.3239
#>  -0.6041 -0.0217 -0.5199  0.0655 -0.1096
#>   1.5206 -1.4638  0.1619 -0.5683 -0.2868
#> 
#> (6,.,.) = 
#> -1.8957  0.1516  1.3762  1.6658 -0.8493
#>   0.9377  2.0748 -0.7561 -1.7704  0.9138
#>   0.2858 -2.6869  0.9798  1.6752 -0.5037
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
