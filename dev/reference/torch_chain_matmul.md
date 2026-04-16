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
#>  -3.9832  14.1464   9.5566   7.6407   4.9891 -16.7007 -19.9746
#>   5.1362   1.3085  -5.6541  -0.7398   0.2510  -0.8794   0.7923
#>   8.5838   2.9096   2.5785  11.5352  -1.9769  -6.1801  -9.3267
#> [ CPUFloatType{3,7} ]
```
