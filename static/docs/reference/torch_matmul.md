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
#>  3.1620 -0.4859 -0.4845  3.9137 -0.1921
#>  -1.6485  0.2956  0.6097 -3.7480 -0.2699
#>  -0.3491  0.5139 -1.5945  0.5275  0.4684
#> 
#> (2,.,.) = 
#> -0.3213 -0.5909  0.2045  0.4425  1.7471
#>   2.1296 -0.6769 -0.6496  0.9591  1.5577
#>   2.9631 -0.0211 -1.1251  1.7944 -0.5125
#> 
#> (3,.,.) = 
#>  1.7979  0.8839 -3.1795  3.7068 -0.3955
#>   1.6851 -0.0341  1.0070  1.4435 -2.2751
#>  -2.1910 -0.0569 -0.7153 -0.7926  2.4281
#> 
#> (4,.,.) = 
#>  0.5425 -0.7263  1.6610  2.5177 -0.3472
#>  -1.0263 -0.3434  1.7177 -0.5433 -0.3449
#>  -1.3352 -0.4591  2.0788 -1.8036 -0.0931
#> 
#> (5,.,.) = 
#> -0.5178  0.8115 -3.0232 -6.3527  2.1662
#>   0.6399  0.0538  1.6850 -0.3027 -2.5606
#>  -1.1840 -0.2581  0.9552 -1.7396  0.5458
#> 
#> (6,.,.) = 
#>  0.6430 -0.7078  1.5422 -0.6773  0.0499
#>   1.2716 -0.8089 -0.3255  0.0915  2.1827
#>   1.9507  0.5033 -1.5173  2.0985 -1.1248
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
