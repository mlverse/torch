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
#>  4.4402  0.4794  3.7050 -4.0910 -1.5955
#>  -2.1442 -0.8195  1.4882  0.3810 -1.9298
#>   0.4326  3.6017 -0.3964 -0.1175  3.1404
#> 
#> (2,.,.) = 
#> -3.1015 -0.5875  1.9961  0.2856 -1.2631
#>   0.8323 -1.3825 -0.9898  0.0486 -0.1543
#>   2.0856  1.2844 -0.5411 -0.4204  0.4489
#> 
#> (3,.,.) = 
#>  1.3630 -1.1036  1.4807 -1.4718 -1.4708
#>   1.2641  0.3542  2.9198 -2.7558  0.3931
#>   1.1566 -0.7938 -3.1348  1.9752 -1.2569
#> 
#> (4,.,.) = 
#>  1.6856 -1.0398 -0.7342 -0.6340  0.3800
#>  -0.8524  0.3108 -2.6837  2.6914 -0.7931
#>   1.2712  0.1029  0.0895 -0.8952  0.7688
#> 
#> (5,.,.) = 
#> -1.5979 -0.3677 -0.7521  1.1732  0.0782
#>  -2.7229  0.5703  2.3275 -0.4265  0.4165
#>   0.8708  1.7878  1.1219 -0.8413  0.1621
#> 
#> (6,.,.) = 
#> -1.9066  2.1288  3.5536 -1.7664  1.7233
#>   0.8327  2.6699  2.3286 -1.6376  0.6059
#>  -3.5105 -1.5770  0.1269  1.8879 -2.1251
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
