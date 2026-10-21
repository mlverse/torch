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
#> -0.3039  0.8110  1.6312 -1.0371  1.8727
#>   0.8928  0.7617  1.4915  0.4579 -2.5973
#>  -0.6341 -1.1168 -3.8346  0.6619  1.6510
#> 
#> (2,.,.) = 
#>  0.0763  1.6965 -1.3277  0.5935  2.0829
#>   0.0584  0.1100  0.8693  0.0938 -0.3938
#>   0.1387  2.1681 -1.9288  2.6900  1.7426
#> 
#> (3,.,.) = 
#>  1.1354  1.5703  3.2528 -0.3940 -2.5843
#>   0.6679  0.2684 -1.5122  0.6315 -1.5295
#>   2.0644  3.4699 -2.5018  3.0284 -2.6654
#> 
#> (4,.,.) = 
#> -1.6887 -2.6890  0.2430 -1.0859  2.3728
#>   0.8506 -0.2046 -1.3651  1.8177 -3.2862
#>   1.3215  2.1187 -1.4061  1.5340 -1.7209
#> 
#> (5,.,.) = 
#>  1.4931  2.9306 -4.1687  2.1880 -0.5720
#>   0.4785 -0.6415  0.5585  0.1618 -2.5853
#>  -0.1515 -0.3505 -1.8062  0.7710  0.3254
#> 
#> (6,.,.) = 
#>  0.1021  0.3077  0.0460  1.5023 -0.6301
#>   0.4987  1.1295  2.4789 -0.4652 -0.8206
#>   1.0754  2.0410 -2.6020  2.5299 -1.0698
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
