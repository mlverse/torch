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
#>   1.5370 -1.9865
#>  -0.2046  0.2645
#> 
#> (2,.,.) = 
#>  0.01 *
#>  -1.3050 -0.2478
#>    0.0217 -1.3725
#> [ CPUFloatType{2,2,2} ]
```
