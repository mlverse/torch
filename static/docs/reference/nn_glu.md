# GLU module

Applies the gated linear unit function \\{GLU}(a, b)= a \otimes
\sigma(b)\\ where \\a\\ is the first half of the input matrices and
\\b\\ is the second half.

## Usage

``` r
nn_glu(dim = -1)
```

## Arguments

- dim:

  (int): the dimension on which to split the input. Default: -1

## Shape

- Input: \\(\ast_1, N, \ast_2)\\ where `*` means, any number of
  additional dimensions

- Output: \\(\ast_1, M, \ast_2)\\ where \\M=N/2\\

## Examples

``` r
if (torch_is_installed()) {
m <- nn_glu()
input <- torch_randn(4, 2)
output <- m(input)
}
```
