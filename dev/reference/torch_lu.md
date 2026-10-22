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
#> -0.8497  0.2888 -0.2846
#>   0.3982  0.9506 -0.3910
#>  -0.9626 -0.4155 -0.2000
#> 
#> (2,.,.) = 
#>  0.5130  0.1899 -0.0197
#>   0.5475 -0.6794 -0.9336
#>   0.0149 -0.2255 -1.5090
#> [ CPUFloatType{2,3,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  3  3  3
#>  2  3  3
#> [ CPUIntType{2,3} ]
#> 
```
