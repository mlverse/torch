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
#>  0.0320  3.1097 -0.2390  0.8977 -1.8805
#>   0.3716 -1.6132 -0.3065 -0.7837  0.7476
#>  -0.7981 -0.6069  0.3805  0.3223 -7.3491
#> 
#> (2,.,.) = 
#>  0.5877  1.0807 -0.0278  0.0676  5.0585
#>  -2.1806 -2.7094 -0.3947 -0.1927  1.1117
#>  -1.9873 -4.0317  0.6713 -0.2390 -2.0332
#> 
#> (3,.,.) = 
#>  1.4278  1.4636  1.0275  0.3420 -1.8543
#>   0.0314 -1.2150  0.2412 -0.3280  3.1550
#>  -0.3204 -0.2338  1.1575  0.5244 -1.9976
#> 
#> (4,.,.) = 
#> -0.3108 -1.1794  0.3468 -0.0805 -4.7862
#>  -2.5263  2.9462  0.1157  1.9649 -1.6221
#>   0.2451 -0.7693  0.1235 -0.2804 -1.1700
#> 
#> (5,.,.) = 
#>  0.5278 -4.5107  1.3610 -1.0493 -6.0407
#>   0.5720 -2.4903  0.5408 -0.7696 -3.4445
#>   0.6495  1.2586  1.3952  0.7735 -8.4446
#> 
#> (6,.,.) = 
#> -0.5651  1.9082 -0.3089  0.6536  5.9992
#>  -1.6911  0.6294  0.4723  1.0297  0.8135
#>   1.1409 -0.7942  1.4108 -0.1053 -3.0062
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
