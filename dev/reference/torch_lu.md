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
#>  1.8313  0.7467 -1.0404
#>   0.5378  0.3569  0.8120
#>   0.1187  0.9366 -1.1194
#> 
#> (2,.,.) = 
#>  0.6839 -0.7574 -0.9158
#>   0.4750  1.8896 -0.6602
#>  -0.7740 -0.2395 -1.1813
#> [ CPUFloatType{2,3,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  2  3  3
#>  1  2  3
#> [ CPUIntType{2,3} ]
#> 
```
