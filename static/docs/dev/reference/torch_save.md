# Saves an object to a disk file.

This function is experimental, don't use for long term storage.

## Usage

``` r
torch_save(obj, path, ..., compress = TRUE)
```

## Arguments

- obj:

  the saved object

- path:

  a connection or the name of the file to save.

- ...:

  not currently used.

- compress:

  a logical specifying whether saving to a named file is to use "gzip"
  compression, or one of "gzip", "bzip2" or "xz" to indicate the type of
  compression to be used. Ignored if file is a connection.

## See also

Other torch_save:
[`torch_load()`](https://torch.mlverse.org/docs/dev/reference/torch_load.md),
[`torch_serialize()`](https://torch.mlverse.org/docs/dev/reference/torch_serialize.md)
