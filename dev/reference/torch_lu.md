# LU

Computes the LU factorization of a matrix or batches of matrices A.
Returns a tuple containing the LU factorization and pivots of A.
Pivoting is done if pivot is set to True.

## Usage

``` r
torch_lu(A, pivot = TRUE, get_infos = FALSE, out = NULL)
```

## Arguments

- A:

  (Tensor) the tensor to factor of size (*, m, n)(*,m,n)

- pivot:

  (bool, optional) – controls whether pivoting is done. Default: TRUE

- get_infos:

  (bool, optional) – if set to True, returns an info IntTensor. Default:
  FALSE

- out:

  (tuple, optional) – optional output tuple. If get_infos is True, then
  the elements in the tuple are Tensor, IntTensor, and IntTensor. If
  get_infos is False, then the elements in the tuple are Tensor,
  IntTensor. Default: NULL

## Examples

``` r
if (torch_is_installed()) {

A <- torch_randn(c(2, 3, 3))
torch_lu(A)
}
#> [[1]]
#> torch_tensor
#> (1,.,.) = 
#>  -0.4367 -0.1352  0.7028
#>  -0.1170 -0.4860 -0.4949
#>   0.2390 -0.8457 -0.8974
#> 
#> (2,.,.) = 
#>  -1.3535 -2.1193  0.5161
#>  -0.3863  1.2242  1.2239
#>  -0.3354 -0.2187  0.1799
#> [ CPUFloatType{2,3,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  2  3  3
#>  1  2  3
#> [ CPUIntType{2,3} ]
#> 
```
