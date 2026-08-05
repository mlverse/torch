# Pdist

Pdist

## Usage

``` r
torch_pdist(self, p = 2L)
```

## Arguments

- self:

  NA input tensor of shape \\N \times M\\.

- p:

  NA p value for the p-norm distance to calculate between each vector
  pair \\\in \[0, \infty\]\\.

## pdist(input, p=2) -\> Tensor

Computes the p-norm distance between every pair of row vectors in the
input. This is identical to the upper triangular portion, excluding the
diagonal, of `torch_norm(input[:, NULL] - input, dim=2, p=p)`. This
function will be faster if the rows are contiguous.

If input has shape \\N \times M\\ then the output will have shape
\\\frac{1}{2} N (N - 1)\\.

This function is equivalent to
`scipy.spatial.distance.pdist(input, 'minkowski', p=p)` if \\p \in (0,
\infty)\\. When \\p = 0\\ it is equivalent to
`scipy.spatial.distance.pdist(input, 'hamming') * M`. When \\p =
\infty\\, the closest scipy function is
`scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.
