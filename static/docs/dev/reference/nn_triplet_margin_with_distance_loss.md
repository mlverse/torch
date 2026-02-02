# Triplet margin with distance loss

Creates a criterion that measures the triplet loss given input tensors
\\a\\, \\p\\, and \\n\\ (representing anchor, positive, and negative
examples, respectively), and a nonnegative, real-valued function
("distance function") used to compute the relationship between the
anchor and positive example ("positive distance") and the anchor and
negative example ("negative distance").

## Usage

``` r
nn_triplet_margin_with_distance_loss(
  distance_function = NULL,
  margin = 1,
  swap = FALSE,
  reduction = "mean"
)
```

## Arguments

- distance_function:

  (callable, optional): A nonnegative, real-valued function that
  quantifies the closeness of two tensors. If not specified,
  [`nn_pairwise_distance()`](https://torch.mlverse.org/docs/dev/reference/nn_pairwise_distance.md)
  will be used. Default: `None`

- margin:

  (float, optional): A non-negative margin representing the minimum
  difference between the positive and negative distances required for
  the loss to be 0. Larger margins penalize cases where the negative
  examples are not distant enough from the anchors, relative to the
  positives. Default: \\1\\.

- swap:

  (bool, optional): Whether to use the distance swap described in the
  paper Learning shallow convolutional feature descriptors with triplet
  losses [doi:10.5244/C.30.119](https://doi.org/10.5244/C.30.119) by V.
  Balntas, E. Riba et al. If TRUE, and if the positive example is closer
  to the negative example than the anchor is, swaps the positive example
  and the anchor in the loss computation. Default: `FALSE`.

- reduction:

  (string, optional): Specifies the (optional) reduction to apply to the
  output: `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will
  be applied, `'mean'`: the sum of the output will be divided by the
  number of elements in the output, `'sum'`: the output will be summed.
  Default: `'mean'`

## Details

The unreduced loss (i.e., with `reduction` set to `'none'`) can be
described as:

\$\$ \ell(a, p, n) = L = \\l_1,\dots,l_N\\^\top, \quad l_i = \max
\\d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\\ \$\$

where \\N\\ is the batch size; \\d\\ is a nonnegative, real-valued
function quantifying the closeness of two tensors, referred to as the
`distance_function`; and \\margin\\ is a non-negative margin
representing the minimum difference between the positive and negative
distances that is required for the loss to be 0. The input tensors have
\\N\\ elements each and can be of any shape that the distance function
can handle. If `reduction` is not `'none'` (default `'mean'`), then:

\$\$ \ell(x, y) = \begin{array}{ll} \mbox{mean}(L), & \mbox{if
reduction} = \mbox{\`mean';}\\ \mbox{sum}(L), & \mbox{if reduction} =
\mbox{\`sum'.} \end{array} \$\$

See also
[`nn_triplet_margin_loss()`](https://torch.mlverse.org/docs/dev/reference/nn_triplet_margin_loss.md),
which computes the triplet loss for input tensors using the \\l_p\\
distance as the distance function.

## Shape

- Input: \\(N, \*)\\ where \\\*\\ represents any number of additional
  dimensions as supported by the distance function.

- Output: A Tensor of shape \\(N)\\ if `reduction` is `'none'`, or a
  scalar otherwise.

## Examples

``` r
if (torch_is_installed()) {
# Initialize embeddings
embedding <- nn_embedding(1000, 128)
anchor_ids <- torch_randint(1, 1000, 1, dtype = torch_long())
positive_ids <- torch_randint(1, 1000, 1, dtype = torch_long())
negative_ids <- torch_randint(1, 1000, 1, dtype = torch_long())
anchor <- embedding(anchor_ids)
positive <- embedding(positive_ids)
negative <- embedding(negative_ids)

# Built-in Distance Function
triplet_loss <- nn_triplet_margin_with_distance_loss(
  distance_function = nn_pairwise_distance()
)
output <- triplet_loss(anchor, positive, negative)

# Custom Distance Function
l_infinity <- function(x1, x2) {
  torch_max(torch_abs(x1 - x2), dim = 1)[[1]]
}

triplet_loss <- nn_triplet_margin_with_distance_loss(
  distance_function = l_infinity, margin = 1.5
)
output <- triplet_loss(anchor, positive, negative)

# Custom Distance Function (Lambda)
triplet_loss <- nn_triplet_margin_with_distance_loss(
  distance_function = function(x, y) {
    1 - nnf_cosine_similarity(x, y)
  }
)

output <- triplet_loss(anchor, positive, negative)
}
```
