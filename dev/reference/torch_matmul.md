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
#>  2.5170  2.6437 -2.6281 -3.7041  1.7071
#>  -1.1157  0.4340  0.6580 -0.4053 -1.5077
#>   2.2303  1.8340 -2.4324 -2.3581  3.4349
#> 
#> (2,.,.) = 
#>  1.7468  1.6894 -1.6987 -2.2406  0.5775
#>  -0.1871  1.0452 -0.3520 -2.1345  1.0154
#>   0.9925  1.1666 -0.8115 -2.0845 -0.9277
#> 
#> (3,.,.) = 
#> -2.4310 -0.9325  1.9641  1.7155 -2.1102
#>  -0.0344  1.7444 -0.8298 -3.1569  2.0083
#>   1.3048 -0.5659 -0.7641  1.5134  1.1690
#> 
#> (4,.,.) = 
#> -2.1704 -0.2495  1.3528 -0.5130  0.3554
#>  -0.3284 -1.6800  0.9859  1.6448 -0.7823
#>   0.6169  1.1771 -0.7599 -1.2667 -0.4293
#> 
#> (5,.,.) = 
#> -2.5462  2.2662  0.4389 -4.8085  2.3528
#>   2.3235 -1.3721 -0.7731  2.2107 -0.4239
#>   2.7951 -0.2133 -1.6949  0.4668  1.3088
#> 
#> (6,.,.) = 
#> -0.0194 -0.1665 -0.0281  0.2054  0.7375
#>  -0.5389  0.2652  0.2017 -0.5224  0.1399
#>   1.5078  0.6424 -1.5891 -0.6924  3.4512
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
