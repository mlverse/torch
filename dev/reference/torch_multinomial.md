# Multinomial

Multinomial

## Usage

``` r
torch_multinomial(self, num_samples, replacement = FALSE, generator = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor containing probabilities

- num_samples:

  (int) number of samples to draw

- replacement:

  (bool, optional) whether to draw with replacement or not

- generator:

  (`torch.Generator`, optional) a pseudorandom number generator for
  sampling

## Note

    The rows of `input` do not need to sum to one (in which case we use
    the values as weights), but must be non-negative, finite and have
    a non-zero sum.

Indices are ordered from left to right according to when each was
sampled (first samples are placed in first column).

If `input` is a vector, `out` is a vector of size `num_samples`.

If `input` is a matrix with `m` rows, `out` is an matrix of shape \\(m
\times \mbox{num\\samples})\\.

If replacement is `TRUE`, samples are drawn with replacement.

If not, they are drawn without replacement, which means that when a
sample index is drawn for a row, it cannot be drawn again for that row.

    When drawn without replacement, `num_samples` must be lower than
    number of non-zero elements in `input` (or the min number of non-zero
    elements in each row of `input` if it is a matrix).

## multinomial(input, num_samples, replacement=False, \*, generator=NULL, out=NULL) -\> LongTensor

Returns a tensor where each row contains `num_samples` indices sampled
from the multinomial probability distribution located in the
corresponding row of tensor `input`.

## Examples

``` r
if (torch_is_installed()) {

weights = torch_tensor(c(0, 10, 3, 0), dtype=torch_float()) # create a tensor of weights
torch_multinomial(weights, 2)
torch_multinomial(weights, 4, replacement=TRUE)
}
#> torch_tensor
#>  2
#>  3
#>  2
#>  2
#> [ CPULongType{4} ]
```
