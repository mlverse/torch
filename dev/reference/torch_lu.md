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
#>  2.5287  0.2450 -0.5817
#>  -0.5854  1.5795 -2.3380
#>   0.3288  0.3886  1.0806
#> 
#> (2,.,.) = 
#> -0.8829 -0.0908 -0.4357
#>  -0.6428  0.9982  1.0794
#>   0.7407 -0.7713  2.3319
#> [ CPUFloatType{2,3,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  3  2  3
#>  2  3  3
#> [ CPUIntType{2,3} ]
#> 
```
