# Serialize a torch object returning a raw object

It's just a wraper around
[`torch_save()`](https://torch.mlverse.org/docs/dev/reference/torch_save.md).

## Usage

``` r
torch_serialize(obj, ...)
```

## Arguments

- obj:

  the saved object

- ...:

  Additional arguments passed to
  [`torch_save()`](https://torch.mlverse.org/docs/dev/reference/torch_save.md).
  `obj` and `path` are not accepted as they are set by
  `torch_serialize()`.

## Value

A raw vector containing the serialized object. Can be reloaded using
[`torch_load()`](https://torch.mlverse.org/docs/dev/reference/torch_load.md).

## See also

Other torch_save:
[`torch_load()`](https://torch.mlverse.org/docs/dev/reference/torch_load.md),
[`torch_save()`](https://torch.mlverse.org/docs/dev/reference/torch_save.md)
