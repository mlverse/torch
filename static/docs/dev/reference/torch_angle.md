# Angle

Angle

## Usage

``` r
torch_angle(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## angle(input) -\> Tensor

Computes the element-wise angle (in radians) of the given `input`
tensor.

\$\$ \mbox{out}\_{i} = angle(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
torch_angle(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))*180/3.14159
} # }

}
```
