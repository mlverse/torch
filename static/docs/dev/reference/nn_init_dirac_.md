# Dirac initialization

Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta
function. Preserves the identity of the inputs in `Convolutional`
layers, where as many input channels are preserved as possible. In case
of groups\>1, each group of channels preserves identity.

## Usage

``` r
nn_init_dirac_(tensor, groups = 1)
```

## Arguments

- tensor:

  a {3, 4, 5}-dimensional `torch.Tensor`

- groups:

  (optional) number of groups in the conv layer (default: 1)

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
w <- torch_empty(3, 16, 5, 5)
nn_init_dirac_(w)
} # }

}
```
