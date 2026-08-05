# Conj

Conj

## Usage

``` r
torch_conj(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## conj(input) -\> Tensor

Computes the element-wise conjugate of the given `input` tensor.

\$\$ \mbox{out}\_{i} = conj(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
torch_conj(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
} # }
}
```
