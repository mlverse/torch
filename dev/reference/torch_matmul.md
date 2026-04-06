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
#>  2.3555  4.2792 -0.5651 -1.3630 -1.0186
#>   1.4321  1.1150 -3.1227 -1.9201  0.3568
#>  -1.5823 -1.7317  0.1589  0.9485  0.1204
#> 
#> (2,.,.) = 
#>  1.9236  1.2755 -0.2878 -0.6341  0.1798
#>  -2.0470 -3.0913  1.0677  2.2304  0.3908
#>   0.3489  1.9999  2.0511  0.2082 -0.9408
#> 
#> (3,.,.) = 
#> -4.5469 -3.9322  3.0344  2.9899 -0.2715
#>  -2.3105 -3.1429  3.1438  2.4187  0.2421
#>  -0.5158 -0.3326  0.7998  0.8560 -0.1821
#> 
#> (4,.,.) = 
#>  0.2296 -2.0672  0.6570 -0.2777  1.1331
#>  -0.1111 -0.6100 -0.3862  0.0732  0.2573
#>   2.6625  5.6039 -1.6307 -1.4987 -1.5136
#> 
#> (5,.,.) = 
#> -0.8720  0.7355  2.3211  0.7938 -0.8627
#>  -0.9397 -3.2869 -1.0904  1.0041  1.1400
#>  -0.0887 -0.1576 -0.9687 -0.5708  0.1615
#> 
#> (6,.,.) = 
#> -2.5705 -4.5389  2.1604  1.6604  0.9831
#>   1.0237  2.5812 -2.4725 -1.4984 -0.5985
#>   1.0790  2.2285 -1.2220 -0.4954 -0.5980
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
