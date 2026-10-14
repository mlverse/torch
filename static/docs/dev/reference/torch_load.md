# Loads a saved object

Loads a saved object

## Usage

``` r
torch_load(path, device = "cpu")
```

## Arguments

- path:

  a path to the saved object

- device:

  a device to load tensors to. By default we load to the `cpu` but you
  can also load them to any `cuda` device. If `NULL` then the device
  where the tensor has been saved will be reused.

## See also

Other torch_save:
[`torch_save()`](https://torch.mlverse.org/docs/dev/reference/torch_save.md),
[`torch_serialize()`](https://torch.mlverse.org/docs/dev/reference/torch_serialize.md)
