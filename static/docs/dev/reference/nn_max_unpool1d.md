# Computes a partial inverse of `MaxPool1d`.

`MaxPool1d` is not fully invertible, since the non-maximal values are
lost. `MaxUnpool1d` takes in as input the output of `MaxPool1d`
including the indices of the maximal values and computes a partial
inverse in which all non-maximal values are set to zero.

## Usage

``` r
nn_max_unpool1d(kernel_size, stride = NULL, padding = 0)
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

`MaxPool1d` can map several input sizes to the same output sizes. Hence,
the inversion process can get ambiguous. To accommodate this, you can
provide the needed output size as an additional argument `output_size`
in the forward call. See the Inputs and Example below.

## Inputs

- `input`: the input Tensor to invert

- `indices`: the indices given out by
  [`nn_max_pool1d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_pool1d.md)

- `output_size` (optional): the targeted output size

## Shape

- Input: \\(N, C, H\_{in})\\

- Output: \\(N, C, H\_{out})\\, where \$\$ H\_{out} = (H\_{in} - 1)
  \times \mbox{stride}\[0\] - 2 \times \mbox{padding}\[0\] +
  \mbox{kernel\\size}\[0\] \$\$ or as given by `output_size` in the call
  operator

## Examples

``` r
if (torch_is_installed()) {
pool <- nn_max_pool1d(2, stride = 2, return_indices = TRUE)
unpool <- nn_max_unpool1d(2, stride = 2)

input <- torch_tensor(array(1:8 / 1, dim = c(1, 1, 8)))
out <- pool(input)
unpool(out[[1]], out[[2]])

# Example showcasing the use of output_size
input <- torch_tensor(array(1:8 / 1, dim = c(1, 1, 8)))
out <- pool(input)
unpool(out[[1]], out[[2]], output_size = input$size())
unpool(out[[1]], out[[2]])
}
#> torch_tensor
#> (1,1,.,.) = 
#>   0
#>   2
#>   0
#>   4
#>   0
#>   6
#>   0
#>   8
#> [ CPUFloatType{1,1,8,1} ]
```
