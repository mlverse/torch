# Vander

Vander

## Usage

``` r
torch_vander(x, N = NULL, increasing = FALSE)
```

## Arguments

- x:

  (Tensor) 1-D input tensor.

- N:

  (int, optional) Number of columns in the output. If N is not
  specified, a square array is returned \\(N = len(x))\\.

- increasing:

  (bool, optional) Order of the powers of the columns. If TRUE, the
  powers increase from left to right, if FALSE (the default) they are
  reversed.

## vander(x, N=None, increasing=FALSE) -\> Tensor

Generates a Vandermonde matrix.

The columns of the output matrix are elementwise powers of the input
vector \\x^{(N-1)}, x^{(N-2)}, ..., x^0\\. If increasing is TRUE, the
order of the columns is reversed \\x^0, x^1, ..., x^{(N-1)}\\. Such a
matrix with a geometric progression in each row is named for
Alexandre-Theophile Vandermonde.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_tensor(c(1, 2, 3, 5))
torch_vander(x)
torch_vander(x, N=3)
torch_vander(x, N=3, increasing=TRUE)
}
#> torch_tensor
#>   1   1   1
#>   1   2   4
#>   1   3   9
#>   1   5  25
#> [ CPUFloatType{4,3} ]
```
