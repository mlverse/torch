# Lu_unpack

Lu_unpack

## Usage

``` r
torch_lu_unpack(LU_data, LU_pivots, unpack_data = TRUE, unpack_pivots = TRUE)
```

## Arguments

- LU_data:

  (Tensor) – the packed LU factorization data

- LU_pivots:

  (Tensor) – the packed LU factorization pivots

- unpack_data:

  (logical) – flag indicating if the data should be unpacked. If FALSE,
  then the returned L and U are NULL Default: TRUE

- unpack_pivots:

  (logical) – flag indicating if the pivots should be unpacked into a
  permutation matrix P. If FALSE, then the returned P is None. Default:
  TRUE

## lu_unpack(LU_data, LU_pivots, unpack_data = TRUE, unpack_pivots=TRUE) -\> Tensor

Unpacks the data and pivots from a LU factorization of a tensor into
tensors `L` and `U` and a permutation tensor `P` such that
`LU_data_and_pivots <- torch_lu(P$matmul(L)$matmul(U))`. Returns a list
of tensors as
`list(the P tensor (permutation matrix), the L tensor, the U tensor)`
