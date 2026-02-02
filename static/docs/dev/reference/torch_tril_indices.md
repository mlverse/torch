# Tril_indices

Tril_indices

## Usage

``` r
torch_tril_indices(
  row,
  col,
  offset = 0,
  dtype = NULL,
  device = NULL,
  layout = NULL
)
```

## Arguments

- row:

  (`int`) number of rows in the 2-D matrix.

- col:

  (`int`) number of columns in the 2-D matrix.

- offset:

  (`int`) diagonal offset from the main diagonal. Default: if not
  provided, 0.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor.
  Default: if `NULL`, `torch_long`.

- device:

  (`torch.device`, optional) the desired device of returned tensor.
  Default: if `NULL`, uses the current device for the default tensor
  type (see `torch_set_default_tensor_type`). `device` will be the CPU
  for CPU tensor types and the current CUDA device for CUDA tensor
  types.

- layout:

  (`torch.layout`, optional) currently only support `torch_strided`.

## Note

    When running on CUDA, `row * col` must be less than \eqn{2^{59}} to
    prevent overflow during calculation.

## tril_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) -\> Tensor

Returns the indices of the lower triangular part of a `row`-by- `col`
matrix in a 2-by-N Tensor, where the first row contains row coordinates
of all indices and the second row contains column coordinates. Indices
are ordered based on rows and then columns.

The lower triangular part of the matrix is defined as the elements on
and below the diagonal.

The argument `offset` controls which diagonal to consider. If `offset` =
0, all elements on and below the main diagonal are retained. A positive
value includes just as many diagonals above the main diagonal, and
similarly a negative value excludes just as many diagonals below the
main diagonal. The main diagonal are the set of indices \\\lbrace (i, i)
\rbrace\\ for \\i \in \[0, \min\\d\_{1}, d\_{2}\\ - 1\]\\ where
\\d\_{1}, d\_{2}\\ are the dimensions of the matrix.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
a = torch_tril_indices(3, 3)
a
a = torch_tril_indices(4, 3, -1)
a
a = torch_tril_indices(4, 3, 1)
a
} # }
}
```
