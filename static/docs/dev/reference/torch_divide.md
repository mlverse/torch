# Divide

Divide

## Usage

``` r
torch_divide(self, other, rounding_mode)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Number) the number to be divided to each element of `input`

- rounding_mode:

  (str, optional) – Type of rounding applied to the result:

  - `NULL` - default behavior. Performs no rounding and, if both input
    and other are integer types, promotes the inputs to the default
    scalar type. Equivalent to true division in Python (the / operator)
    and NumPy’s `np.true_divide`.

  - "trunc" - rounds the results of the division towards zero.
    Equivalent to C-style integer division.

  - "floor" - rounds the results of the division down. Equivalent to
    floor division in Python (the // operator) and NumPy’s
    `np.floor_divide`.

## divide(input, other, \*, out=None) -\> Tensor

Alias for
[`torch_div()`](https://torch.mlverse.org/docs/dev/reference/torch_div.md).
