# Pairwise distance

Computes the batchwise pairwise distance between vectors \\v_1\\,
\\v_2\\ using the p-norm:

## Usage

``` r
nn_pairwise_distance(p = 2, eps = 1e-06, keepdim = FALSE)
```

## Arguments

- p:

  (real): the norm degree. Default: 2

- eps:

  (float, optional): Small value to avoid division by zero. Default:
  1e-6

- keepdim:

  (bool, optional): Determines whether or not to keep the vector
  dimension. Default: FALSE

## Details

\$\$ \Vert x \Vert \_p = \left( \sum\_{i=1}^n \vert x_i \vert ^ p
\right) ^ {1/p}. \$\$

## Shape

- Input1: \\(N, D)\\ where `D = vector dimension`

- Input2: \\(N, D)\\, same shape as the Input1

- Output: \\(N)\\. If `keepdim` is `TRUE`, then \\(N, 1)\\.

## Examples

``` r
if (torch_is_installed()) {
pdist <- nn_pairwise_distance(p = 2)
input1 <- torch_randn(100, 128)
input2 <- torch_randn(100, 128)
output <- pdist(input1, input2)
}
```
