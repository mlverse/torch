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
#>  -0.9379  1.2176  4.6013 -0.3942  0.5170
#>  -1.3289  0.9840 -0.9148 -1.0966  1.3649
#>   1.1799  0.0049 -0.6406 -0.7679 -1.5093
#> 
#> (2,.,.) = 
#>  -0.8472  2.1566 -4.8498 -0.3049 -3.4023
#>   2.0549  1.2507 -0.2527 -1.2896 -5.0128
#>   0.5014  0.2095 -0.0913 -1.3016 -0.0484
#> 
#> (3,.,.) = 
#>  -1.2022  0.2633 -1.9829  1.3325 -0.0479
#>   0.4817  0.7643 -4.6796 -0.9772 -2.3154
#>  -2.2584  5.6471 -0.3154 -2.5210 -4.7850
#> 
#> (4,.,.) = 
#>  -1.0508  1.3058 -0.1045 -0.8487  0.1157
#>  -0.0457  1.5951 -0.7858  0.6597 -3.9335
#>  -0.0723  0.3588  0.7123  0.0700 -0.5265
#> 
#> (5,.,.) = 
#>  -0.8170  2.1326  3.8205 -1.1477 -0.9365
#>  -1.1110 -2.3301 -0.6065  2.2925  4.3009
#>   0.2184  1.7307 -4.7895 -1.5101 -3.2675
#> 
#> (6,.,.) = 
#>   1.4948 -0.3074  3.6957 -0.8346 -0.5726
#>  -1.9938  0.2292 -5.4506  0.1024  2.0412
#>  -1.1080  0.5555 -3.8712  1.5851 -1.4145
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
