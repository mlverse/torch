# Softplus module

Applies the element-wise function: \$\$ \mbox{Softplus}(x) =
\frac{1}{\beta} \* \log(1 + \exp(\beta \* x)) \$\$

## Usage

``` r
nn_softplus(beta = 1, threshold = 20)
```

## Arguments

- beta:

  the \\\beta\\ value for the Softplus formulation. Default: 1

- threshold:

  values above this revert to a linear function. Default: 20

## Details

SoftPlus is a smooth approximation to the ReLU function and can be used
to constrain the output of a machine to always be positive. For
numerical stability the implementation reverts to the linear function
when \\input \times \beta \> threshold\\.

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_softplus()
input <- torch_randn(2)
output <- m(input)
}
```
