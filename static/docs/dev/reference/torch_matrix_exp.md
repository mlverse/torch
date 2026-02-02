# Matrix_exp

Matrix_exp

## Usage

``` r
torch_matrix_exp(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## matrix_power(input) -\> Tensor

Returns the matrix exponential. Supports batched input. For a matrix
`A`, the matrix exponential is defined as

\$\$ \exp^A = \sum\_{k=0}^\infty A^k / k!. \$\$

The implementation is based on: Bader, P.; Blanes, S.; Casas, F.
Computing the Matrix Exponential with an Optimized Taylor Polynomial
Approximation. Mathematics 2019, 7, 1174.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_randn(c(2, 2, 2))
a[1, , ] <- torch_eye(2, 2)
a[2, , ] <- 2 * torch_eye(2, 2)
a
torch_matrix_exp(a)

x <- torch_tensor(rbind(c(0, pi/3), c(-pi/3, 0)))
x$matrix_exp() # should be [[cos(pi/3), sin(pi/3)], [-sin(pi/3), cos(pi/3)]]
}
#> torch_tensor
#>  0.5000  0.8660
#> -0.8660  0.5000
#> [ CPUFloatType{2,2} ]
```
