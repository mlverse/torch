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
#> -1.3079  1.7026 -2.4192 -1.0196  0.4392
#>  -0.5123  1.0678 -1.3291 -0.5467  0.9746
#>  -2.4326  3.0766 -3.1775 -0.9151 -1.4682
#> 
#> (2,.,.) = 
#>  0.2005 -3.3015 -0.2934  0.4343 -2.7965
#>   0.5654  0.9062  0.6869 -0.3791  2.6766
#>  -0.2801  4.1647 -0.4277 -0.6673  4.0984
#> 
#> (3,.,.) = 
#>  0.4987 -2.2411  0.6345  0.1137 -1.0313
#>  -1.0005 -0.0434 -2.2988 -1.0633 -0.1974
#>   1.7901 -0.5935  4.4038  1.9145 -0.4436
#> 
#> (4,.,.) = 
#>  1.0248 -1.3226  1.4295  0.4994  0.3768
#>   0.0210  2.0041  1.0759  0.3313  0.7411
#>  -1.5942  5.3962 -2.4062 -1.7490  3.9167
#> 
#> (5,.,.) = 
#>  1.5301 -2.2402  2.2257  1.2514 -0.4547
#>  -1.0720  4.9748 -1.2609 -0.7302  2.9905
#>   1.1268 -1.9349  2.9668  1.7344 -2.5582
#> 
#> (6,.,.) = 
#> -0.5058 -0.1199 -0.6552 -0.2337 -0.9295
#>   0.8271  2.0063  1.6658  0.5926  2.4871
#>  -2.1600  6.2393 -3.3110 -1.7129  3.4344
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
