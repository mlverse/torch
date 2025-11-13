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
#>   2.1832  1.3125 -0.6478
#>  -0.3890  1.7571 -1.4430
#>  -0.1175  0.1264 -1.1267
#> 
#> (2,.,.) = 
#>  -0.7666 -0.5237 -1.1929
#>   0.3008  3.6656  0.9421
#>   0.5528  0.0811  0.5632
#> [ CPUFloatType{2,3,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  3  3  3
#>  3  3  3
#> [ CPUIntType{2,3} ]
#> 
```
