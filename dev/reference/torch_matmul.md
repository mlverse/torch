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
#>   1.1873 -0.1631  2.9997 -0.3121 -0.2226
#>   1.1633  1.3312  4.7265 -1.5575 -1.1016
#>   0.0576 -3.0030 -2.3705  0.5495  1.3227
#> 
#> (2,.,.) = 
#>  -0.6139 -2.2040 -2.3118 -0.2442  0.3142
#>  -0.9111 -4.0994 -4.1348  1.5882  0.3586
#>  -1.4435  2.8874 -2.6553  0.2168  0.1209
#> 
#> (3,.,.) = 
#>  -0.6315  1.8188  1.3507 -0.6726 -1.8564
#>   0.3182 -0.7181  0.1714 -1.0140  0.5850
#>   0.4993 -0.6581  0.7039  3.6944 -0.7764
#> 
#> (4,.,.) = 
#>  -0.6966 -0.3350 -3.1950  0.3739  1.2584
#>   0.2737 -5.4071 -4.5726  3.8004  2.2461
#>  -1.3840 -0.4320 -4.8929  0.0802  1.4305
#> 
#> (5,.,.) = 
#>  -0.4274  0.3204 -0.5727 -2.3142  0.3460
#>  -0.1177  0.1015  0.1356 -1.3779  0.0344
#>  -0.5002 -3.4907 -4.2773  2.7450  1.1759
#> 
#> (6,.,.) = 
#>  -0.1876  0.9511  0.9177 -0.9404 -0.7097
#>  -0.1286  1.6644  0.5388  0.5742 -0.4708
#>  -0.6393  2.6078 -0.3017  2.9460 -1.1560
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
