# Computes a partial inverse of `MaxPool3d`.

`MaxPool3d` is not fully invertible, since the non-maximal values are
lost. `MaxUnpool3d` takes in as input the output of `MaxPool3d`
including the indices of the maximal values and computes a partial
inverse in which all non-maximal values are set to zero.

## Usage

``` r
nn_max_unpool3d(kernel_size, stride = NULL, padding = 0)
```

## Arguments

- kernel_size:

  (int or tuple): Size of the max pooling window.

- stride:

  (int or tuple): Stride of the max pooling window. It is set to
  `kernel_size` by default.

- padding:

  (int or tuple): Padding that was added to the input

## Note

`MaxPool3d` can map several input sizes to the same output sizes. Hence,
the inversion process can get ambiguous. To accommodate this, you can
provide the needed output size as an additional argument `output_size`
in the forward call. See the Inputs section below.

## Inputs

- `input`: the input Tensor to invert

- `indices`: the indices given out by
  [`nn_max_pool3d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_pool3d.md)

- `output_size` (optional): the targeted output size

## Shape

- Input: \\(N, C, D\_{in}, H\_{in}, W\_{in})\\

- Output: \\(N, C, D\_{out}, H\_{out}, W\_{out})\\, where

\$\$ D\_{out} = (D\_{in} - 1) \times \mbox{stride\[0\]} - 2 \times
\mbox{padding\[0\]} + \mbox{kernel\\size\[0\]} \$\$ \$\$ H\_{out} =
(H\_{in} - 1) \times \mbox{stride\[1\]} - 2 \times \mbox{padding\[1\]} +
\mbox{kernel\\size\[1\]} \$\$ \$\$ W\_{out} = (W\_{in} - 1) \times
\mbox{stride\[2\]} - 2 \times \mbox{padding\[2\]} +
\mbox{kernel\\size\[2\]} \$\$

or as given by `output_size` in the call operator

## Examples

``` r
if (torch_is_installed()) {

# pool of square window of size=3, stride=2
pool <- nn_max_pool3d(3, stride = 2, return_indices = TRUE)
unpool <- nn_max_unpool3d(3, stride = 2)
out <- pool(torch_randn(20, 16, 51, 33, 15))
unpooled_output <- unpool(out[[1]], out[[2]])
unpooled_output$size()
}
#> [1] 20 16 51 33 15
```
