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
#>  3.0567 -3.2849 -1.5171  0.5826  1.1343
#>  -1.3697  2.2202  2.1099 -0.0597 -0.7485
#>  -2.7956  3.0727  1.1219 -0.6736 -0.1815
#> 
#> (2,.,.) = 
#> -0.2854  0.6280  1.7163  0.0754 -0.5747
#>  -3.2768  3.9967  1.4756 -0.6906 -0.2132
#>   0.2886 -0.0691  1.7973 -0.0493  0.6795
#> 
#> (3,.,.) = 
#> -6.0878  6.7328  3.0225 -1.1516 -2.0473
#>   2.7670 -2.5380 -0.0679  0.7067  0.4971
#>  -0.8024 -4.4410 -4.5109 -1.7582  1.5417
#> 
#> (4,.,.) = 
#> -1.9386  4.9869  2.6048  0.4531 -1.2812
#>   3.6738  0.4005  2.0654  2.1386 -0.7532
#>  -2.8371 -2.7859 -0.8294 -2.5909  1.9614
#> 
#> (5,.,.) = 
#>  2.2558 -5.7779 -2.9658 -0.5175  1.4651
#>   2.3382 -3.9410 -0.9786  0.0683  0.8564
#>  -3.1430  2.4410 -0.6839 -0.8277 -0.9184
#> 
#> (6,.,.) = 
#>  0.6398  1.6429  2.8400  0.9541 -1.3719
#>  -2.3600  5.4964  4.2048  0.4274 -1.8492
#>  -0.3333 -1.1163 -1.7471 -0.3605 -0.2988
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
