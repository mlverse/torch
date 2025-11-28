# Computes a partial inverse of `MaxPool2d`.

`MaxPool2d` is not fully invertible, since the non-maximal values are
lost. `MaxUnpool2d` takes in as input the output of `MaxPool2d`
including the indices of the maximal values and computes a partial
inverse in which all non-maximal values are set to zero.

## Usage

``` r
nn_max_unpool2d(kernel_size, stride = NULL, padding = 0)
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

`MaxPool2d` can map several input sizes to the same output sizes. Hence,
the inversion process can get ambiguous. To accommodate this, you can
provide the needed output size as an additional argument `output_size`
in the forward call. See the Inputs and Example below.

## Inputs

- `input`: the input Tensor to invert

- `indices`: the indices given out by
  [`nn_max_pool2d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_pool2d.md)

- `output_size` (optional): the targeted output size

## Shape

- Input: \\(N, C, H\_{in}, W\_{in})\\

- Output: \\(N, C, H\_{out}, W\_{out})\\, where \$\$ H\_{out} =
  (H\_{in} - 1) \times \mbox{stride\[0\]} - 2 \times
  \mbox{padding\[0\]} + \mbox{kernel\\size\[0\]} \$\$ \$\$ W\_{out} =
  (W\_{in} - 1) \times \mbox{stride\[1\]} - 2 \times
  \mbox{padding\[1\]} + \mbox{kernel\\size\[1\]} \$\$ or as given by
  `output_size` in the call operator

## Examples

``` r
if (torch_is_installed()) {

pool <- nn_max_pool2d(2, stride = 2, return_indices = TRUE)
unpool <- nn_max_unpool2d(2, stride = 2)
input <- torch_randn(1, 1, 4, 4)
out <- pool(input)
unpool(out[[1]], out[[2]])

# specify a different output size than input size
unpool(out[[1]], out[[2]], output_size = c(1, 1, 5, 5))
}
#> torch_tensor
#> (1,1,.,.) = 
#>   0.0000  0.0000  0.0000  0.0000  0.0000
#>   0.0022  0.0000  1.0334  0.6422  0.0000
#>   0.0000  0.4223  0.0000  0.0000  0.0000
#>   0.0000  0.0000  0.0000  0.0000  0.0000
#>   0.0000  0.0000  0.0000  0.0000  0.0000
#> [ CPUFloatType{1,1,5,5} ]
```
