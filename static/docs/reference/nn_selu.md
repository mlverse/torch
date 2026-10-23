# SELU module

Applied element-wise, as:

## Usage

``` r
nn_selu(inplace = FALSE)
```

## Arguments

- inplace:

  (bool, optional): can optionally do the operation in-place. Default:
  `FALSE`

## Details

\$\$ \mbox{SELU}(x) = \mbox{scale} \* (\max(0,x) + \min(0, \alpha \*
(\exp(x) - 1))) \$\$

with \\\alpha = 1.6732632423543772848170429916717\\ and \\\mbox{scale} =
1.0507009873554804934193349852946\\.

More details can be found in the paper [Self-Normalizing Neural
Networks](https://arxiv.org/abs/1706.02515).

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_selu()
input <- torch_randn(2)
output <- m(input)
}
```
