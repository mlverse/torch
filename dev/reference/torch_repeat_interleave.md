# Repeat_interleave

Repeat_interleave

## Usage

``` r
torch_repeat_interleave(self, repeats, dim = NULL, output_size = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- repeats:

  (Tensor or int) The number of repetitions for each element. repeats is
  broadcasted to fit the shape of the given axis.

- dim:

  (int, optional) The dimension along which to repeat values. By
  default, use the flattened input array, and return a flat output
  array.

- output_size:

  (int, optional) – Total output size for the given axis ( e.g. sum of
  repeats). If given, it will avoid stream syncronization needed to
  calculate output shape of the tensor.

## repeat_interleave(input, repeats, dim=NULL) -\> Tensor

Repeat elements of a tensor.

## Warning

    This is different from `torch_Tensor.repeat` but similar to `numpy.repeat`.

## repeat_interleave(repeats) -\> Tensor

If the `repeats` is `tensor([n1, n2, n3, ...])`, then the output will be
`tensor([0, 0, ..., 1, 1, ..., 2, 2, ..., ...])` where `0` appears `n1`
times, `1` appears `n2` times, `2` appears `n3` times, etc.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
x = torch_tensor(c(1, 2, 3))
x$repeat_interleave(2)
y = torch_tensor(matrix(c(1, 2, 3, 4), ncol = 2, byrow=TRUE))
torch_repeat_interleave(y, 2)
torch_repeat_interleave(y, 3, dim=1)
torch_repeat_interleave(y, torch_tensor(c(1, 2)), dim=1)
} # }
}
```
