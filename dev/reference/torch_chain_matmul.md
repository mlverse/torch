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
#>  -0.8758  15.4466   2.0966   0.4399 -12.8302  -3.7554  -6.8370
#> -12.8183  12.0738   0.4887  15.5417  -1.0073   2.4059   2.1687
#>   7.6522 -13.2436   0.1896  -7.8384   6.5195  -1.4045   2.0287
#> [ CPUFloatType{3,7} ]
```
