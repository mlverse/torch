# RReLU module

Applies the randomized leaky rectified liner unit function,
element-wise, as described in the paper:

## Usage

``` r
nn_rrelu(lower = 1/8, upper = 1/3, inplace = FALSE)
```

## Arguments

- lower:

  lower bound of the uniform distribution. Default: \\\frac{1}{8}\\

- upper:

  upper bound of the uniform distribution. Default: \\\frac{1}{3}\\

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

`Empirical Evaluation of Rectified Activations in Convolutional Network`.

The function is defined as:

\$\$ \mbox{RReLU}(x) = \left\\ \begin{array}{ll} x & \mbox{if } x \geq 0
\\ ax & \mbox{ otherwise } \end{array} \right. \$\$

where \\a\\ is randomly sampled from uniform distribution
\\\mathcal{U}(\mbox{lower}, \mbox{upper})\\. See:
https://arxiv.org/pdf/1505.00853.pdf

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_rrelu(0.1, 0.3)
input <- torch_randn(2)
m(input)
}
#> torch_tensor
#> 0.01 *
#> -8.4270
#> -7.8178
#> [ CPUFloatType{2} ]
```
