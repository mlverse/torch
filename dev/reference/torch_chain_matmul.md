# Chain_matmul

Chain_matmul

## Usage

``` r
torch_chain_matmul(matrices)
```

## Arguments

- matrices:

  (Tensors...) a sequence of 2 or more 2-D tensors whose product is to
  be determined.

## TEST

Returns the matrix product of the \\N\\ 2-D tensors. This product is
efficiently computed using the matrix chain order algorithm which
selects the order in which incurs the lowest cost in terms of arithmetic
operations (`[CLRS]`\_). Note that since this is a function to compute
the product, \\N\\ needs to be greater than or equal to 2; if equal to 2
then a trivial matrix-matrix product is returned. If \\N\\ is 1, then
this is a no-op - the original matrix is returned as is.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3, 4))
b = torch_randn(c(4, 5))
c = torch_randn(c(5, 6))
d = torch_randn(c(6, 7))
torch_chain_matmul(list(a, b, c, d))
}
#> torch_tensor
#> -2.2964  0.1279  0.9731 -0.6632 -1.4185 -1.5069  1.2610
#> -5.2288  9.0646 -10.2009 -4.1643 -2.6269 -7.3602 -2.1186
#>  1.8903  8.1619 -9.0372 -8.1041 -1.1485 -3.9662 -5.6282
#> [ CPUFloatType{3,7} ]
```
