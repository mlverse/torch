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
#> -0.9117  2.1281 -1.5132 -4.9316  3.1746
#>   1.2959 -0.2579 -1.7591  0.3750 -1.8217
#>  -3.2780  2.0415  6.6575  4.4201 -0.4146
#> 
#> (2,.,.) = 
#>  0.4703 -0.6271 -1.0107  0.0108  0.1194
#>  -0.6183  1.8013 -0.1052 -1.7539  0.5559
#>   0.5993 -3.4591 -1.3536  0.3708  2.5367
#> 
#> (3,.,.) = 
#>  1.2677 -2.0925 -2.4892 -1.0202  1.0286
#>   1.7616  0.6874 -0.6178  0.0585 -4.3111
#>  -2.1689  2.5330  1.8170 -1.7417  2.2406
#> 
#> (4,.,.) = 
#> -1.6771  1.3784  0.0841 -0.5948  2.7318
#>   1.4252  0.2603 -2.2357 -1.6510 -1.4147
#>  -2.5679  0.2935  4.0688  2.8467  1.8642
#> 
#> (5,.,.) = 
#> -1.9332 -0.8215  2.2380  1.1666  3.3134
#>  -1.5269  3.3677  3.3727  4.1034 -3.5435
#>  -0.0890  0.5516 -2.5289 -4.2064  3.0707
#> 
#> (6,.,.) = 
#>  1.2275 -3.1747  0.5580  2.8815 -1.4219
#>   0.4386 -0.1556 -0.3302  1.0418 -1.1189
#>  -0.9267  0.9631  1.2130  1.3500 -0.1260
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
