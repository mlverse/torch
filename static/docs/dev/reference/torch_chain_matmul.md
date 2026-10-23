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
#>  -9.7244   1.8069 -11.3284  17.4149  -9.3064  -0.1400  -6.9753
#>   2.8029   0.8529   3.9416  -3.1796   0.7844   3.4570  -1.2846
#>  -0.9804   0.9832  -0.4480   2.5778  -1.7595   2.0343  -3.1726
#> [ CPUFloatType{3,7} ]
```
