# Imag

Imag

## Usage

``` r
torch_imag(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## imag(input) -\> Tensor

Returns the imaginary part of the `input` tensor.

## Warning

Not yet implemented.

\$\$ \mbox{out}\_{i} = imag(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
torch_imag(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
} # }
}
```
