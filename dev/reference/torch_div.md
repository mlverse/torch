# Div

Div

## Usage

``` r
torch_div(self, other, rounding_mode)
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

## div(input, other, out=NULL) -\> Tensor

Divides each element of the input `input` with the scalar `other` and
returns a new resulting tensor.

Each element of the tensor `input` is divided by each element of the
tensor `other`. The resulting tensor is returned.

\$\$ \mbox{out}\_i = \frac{\mbox{input}\_i}{\mbox{other}\_i} \$\$ The
shapes of `input` and `other` must be broadcastable . If the
`torch_dtype` of `input` and `other` differ, the `torch_dtype` of the
result tensor is determined following rules described in the type
promotion documentation . If `out` is specified, the result must be
castable to the `torch_dtype` of the specified output tensor. Integral
division by zero leads to undefined behavior.

## Warning

Integer division using div is deprecated, and in a future release div
will perform true division like
[`torch_true_divide()`](https://torch.mlverse.org/docs/dev/reference/torch_true_divide.md).
Use
[`torch_floor_divide()`](https://torch.mlverse.org/docs/dev/reference/torch_floor_divide.md)
to perform integer division, instead.

\$\$ \mbox{out}\_i = \frac{\mbox{input}\_i}{\mbox{other}} \$\$ If the
`torch_dtype` of `input` and `other` differ, the `torch_dtype` of the
result tensor is determined following rules described in the type
promotion documentation . If `out` is specified, the result must be
castable to the `torch_dtype` of the specified output tensor. Integral
division by zero leads to undefined behavior.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_div(a, 0.5)


a = torch_randn(c(4, 4))
a
b = torch_randn(c(4))
b
torch_div(a, b)
}
#> torch_tensor
#>  0.1251  3.8711  1.2603 -2.1386
#>  0.0559 -1.4412 -0.5739 -0.9099
#>  0.4254 -1.5526  0.5355  0.5755
#> -0.9131  0.1799 -0.3742  0.0743
#> [ CPUFloatType{4,4} ]
```
