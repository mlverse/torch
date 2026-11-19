# Where

Where

## Usage

``` r
torch_where(condition, self = NULL, other = NULL)
```

## Arguments

- condition:

  (BoolTensor) When TRUE (nonzero), yield x, otherwise yield y

- self:

  (Tensor) values selected at indices where `condition` is `TRUE`

- other:

  (Tensor) values selected at indices where `condition` is `FALSE`

## Note

    The tensors `condition`, `x`, `y` must be broadcastable .

See also
[`torch_nonzero()`](https://torch.mlverse.org/docs/dev/reference/torch_nonzero.md).

## where(condition, x, y) -\> Tensor

Return a tensor of elements selected from either `x` or `y`, depending
on `condition`.

The operation is defined as:

\$\$ \mbox{out}\_i = \left\\ \begin{array}{ll} \mbox{x}\_i & \mbox{if }
\mbox{condition}\_i \\ \mbox{y}\_i & \mbox{otherwise} \\ \end{array}
\right. \$\$

## where(condition) -\> tuple of LongTensor

`torch_where(condition)` is identical to
`torch_nonzero(condition, as_tuple=TRUE)`.

## Examples

``` r
if (torch_is_installed()) {

if (FALSE) { # \dontrun{
x = torch_randn(c(3, 2))
y = torch_ones(c(3, 2))
x
torch_where(x > 0, x, y)
} # }



}
```
