# Real

Real

## Usage

``` r
torch_real(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## real(input) -\> Tensor

Returns the real part of the `input` tensor. If `input` is a real
(non-complex) tensor, this function just returns it.

## Warning

Not yet implemented for complex tensors.

\$\$ \mbox{out}\_{i} = real(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
torch_real(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
} # }
}
```
