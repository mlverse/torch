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
#>  -2.4560   6.0927 -25.8179  -4.5028  -1.4661  -3.0200   1.8451
#>   6.2096  -6.9393  23.0657   0.1386   2.7833  -1.3452  -6.1157
#>  -1.4575  -8.1084  23.3537   5.9234  14.1206   2.7327   1.1636
#> [ CPUFloatType{3,7} ]
```
