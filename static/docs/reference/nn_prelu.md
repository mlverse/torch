# PReLU module

Applies the element-wise function: \$\$ \mbox{PReLU}(x) = \max(0,x) + a
\* \min(0,x) \$\$ or \$\$ \mbox{PReLU}(x) = \left\\ \begin{array}{ll} x,
& \mbox{ if } x \geq 0 \\ ax, & \mbox{ otherwise } \end{array} \right.
\$\$

## Usage

``` r
nn_prelu(num_parameters = 1, init = 0.25)
```

## Arguments

- num_parameters:

  (int): number of \\a\\ to learn. Although it takes an int as input,
  there is only two values are legitimate: 1, or the number of channels
  at input. Default: 1

- init:

  (float): the initial value of \\a\\. Default: 0.25

## Details

Here \\a\\ is a learnable parameter. When called without arguments,
`nn.prelu()` uses a single parameter \\a\\ across all input channels. If
called with `nn_prelu(nChannels)`, a separate \\a\\ is used for each
input channel.

## Note

weight decay should not be used when learning \\a\\ for good
performance.

Channel dim is the 2nd dim of input. When input has dims \< 2, then
there is no channel dim and the number of channels = 1.

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Attributes

- weight (Tensor): the learnable weights of shape (`num_parameters`).

## Examples

``` r
if (torch_is_installed()) {
m <- nn_prelu()
input <- torch_randn(2)
output <- m(input)
}
```
