# Create a Device object

A `torch_device` is an object representing the device on which a
`torch_tensor` is or will be allocated.

## Usage

``` r
torch_device(type, index = NULL)
```

## Arguments

- type:

  (character) a device type `"cuda"` or `"cpu"`

- index:

  (integer) optional device ordinal for the device type. If the device
  ordinal is not present, this object will always represent the current
  device for the device type, even after `torch_cuda_set_device()` is
  called; e.g., a `torch_tensor` constructed with device `'cuda'` is
  equivalent to `'cuda:X'` where X is the result of
  `torch_cuda_current_device()`.

  A `torch_device` can be constructed via a string or via a string and
  device ordinal

## Examples

``` r
if (torch_is_installed()) {

# Via string
torch_device("cuda:1")
torch_device("cpu")
torch_device("cuda") # current cuda device

# Via string and device ordinal
torch_device("cuda", 0)
torch_device("cpu", 0)
}
#> torch_device(type='cpu', index=0) 
```
