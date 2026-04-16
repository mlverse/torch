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
#> -1.8891  0.5424 -1.7498 -1.0616 -2.5893
#>  -0.6069 -0.4130  1.5331  0.9377  0.1270
#>   0.7374 -0.9279 -0.8368 -1.2447  1.4836
#> 
#> (2,.,.) = 
#> -1.5334  0.3854  1.1694  2.8642  3.6173
#>   3.9838  4.8021 -3.3727 -1.0479 -6.4434
#>   0.1073  0.8858 -1.0306 -1.2371 -4.0021
#> 
#> (3,.,.) = 
#> -1.1380 -1.0501  0.7346  1.1012  3.8355
#>   2.8034 -3.0281  1.4781 -2.6565  0.0696
#>  -1.6831  1.8021 -1.9453  0.2652 -1.6790
#> 
#> (4,.,.) = 
#>  1.7579  0.0263 -1.7559 -0.9280  2.5570
#>  -1.9439 -0.2943  0.7750  0.3711 -1.4278
#>   1.5929 -2.2165  3.3189 -0.4443 -0.7799
#> 
#> (5,.,.) = 
#>  1.2606 -1.4059 -0.1692 -1.3589  1.5243
#>  -2.3194  0.2596  1.3130  1.8228 -0.1861
#>   0.6246 -0.9340 -1.4547 -4.2957 -6.5016
#> 
#> (6,.,.) = 
#>  1.5603  2.9860 -2.2429 -1.0481 -5.6903
#>   3.0184 -1.7598  0.5778 -1.7750  1.1808
#>   0.0375  2.9664 -4.9181 -2.4862 -5.3686
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
