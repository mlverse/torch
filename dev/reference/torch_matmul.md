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
#> -1.2380 -1.6461  0.4558 -0.3033 -0.9921
#>   3.8751  1.3713 -1.7919  0.9408 -0.1716
#>   0.5924  1.8509  0.9895 -1.0864 -0.4530
#> 
#> (2,.,.) = 
#>  0.0060  0.0966 -0.8680  1.4579  2.0270
#>   4.6878 -1.8382 -3.9888  1.2998 -2.2498
#>   2.1574  3.3713 -2.5799 -1.8122  0.8232
#> 
#> (3,.,.) = 
#> -0.2483  0.5784  1.9295  0.2717 -0.0450
#>  -1.3398  1.4412 -1.7562 -2.3174  1.1258
#>  -0.5393 -2.9360  3.3195  2.2888 -1.7574
#> 
#> (4,.,.) = 
#> -1.6563 -0.8807  2.1589  0.9538  0.3971
#>   1.0981  2.0774 -2.4317 -2.0673  0.2174
#>  -2.4349 -1.9395  1.3827 -0.3382 -0.7687
#> 
#> (5,.,.) = 
#> -1.2937  0.7750  1.2733  0.7846  1.9201
#>   0.7210 -2.7471 -0.7446  0.4866 -2.2471
#>  -1.0563  4.3377 -0.3317 -2.4337  2.6351
#> 
#> (6,.,.) = 
#>  0.2793  0.9693  3.9418 -0.9927 -2.5244
#>   0.6809 -0.3282 -0.2073 -0.3988 -1.1785
#>   0.5296 -1.4662  3.7308  1.8371 -2.0171
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
