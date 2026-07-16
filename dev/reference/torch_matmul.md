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
#> -1.9268 -0.1290  0.3416  0.8652 -1.8294
#>  -0.4310 -1.4549 -0.9116 -0.0464 -0.2522
#>   0.4475  0.9954  0.5673 -0.2792  0.0727
#> 
#> (2,.,.) = 
#>  2.9637 -0.2107  1.4099  0.9341  1.4816
#>   2.3750  0.1675  1.1787  0.9961  1.7245
#>  -4.1420 -0.7073 -1.1798  0.1873 -2.7630
#> 
#> (3,.,.) = 
#> -6.7648  2.0947 -1.7594 -2.6398 -5.1306
#>  -4.5116  2.1055 -1.1686 -1.9637 -2.9721
#>  -4.6646  1.9024 -1.7377 -3.2465 -3.8042
#> 
#> (4,.,.) = 
#>  1.0883  0.3951  0.4732 -0.4429  0.1560
#>  -1.7434  2.8295 -1.1666 -2.6171 -0.0311
#>  -2.9678  0.3383  0.3498 -0.3633 -3.9824
#> 
#> (5,.,.) = 
#> -3.4502  1.3499 -1.9822 -2.5596 -1.7509
#>  -5.8525 -0.1497 -2.2273 -1.8920 -4.5885
#>  -2.2539 -0.3241 -0.2703  0.0216 -2.2374
#> 
#> (6,.,.) = 
#>  2.8622 -0.2450  0.1625  0.2047  2.8082
#>  -0.6754  1.3081 -0.1154 -0.6231  0.0329
#>  -3.6347 -1.0650 -0.3803  0.3493 -3.8736
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
