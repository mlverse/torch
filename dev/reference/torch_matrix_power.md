# Matrix_power

Matrix_power

## Usage

``` r
torch_matrix_power(self, n)
```

## Arguments

- self:

  (Tensor) the input tensor.

- n:

  (int) the power to raise the matrix to

## matrix_power(input, n) -\> Tensor

Returns the matrix raised to the power `n` for square matrices. For
batch of matrices, each individual matrix is raised to the power `n`.

If `n` is negative, then the inverse of the matrix (if invertible) is
raised to the power `n`. For a batch of matrices, the batched inverse
(if invertible) is raised to the power `n`. If `n` is 0, then an
identity matrix is returned.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(2, 2, 2))
a
torch_matrix_power(a, 3)
}
#> torch_tensor
#> (1,.,.) = 
#>  -0.7452 -0.8981
#>  -2.1955 -1.1051
#> 
#> (2,.,.) = 
#>  -0.1528  0.1891
#>  -0.7067  1.2757
#> [ CPUFloatType{2,2,2} ]
```
