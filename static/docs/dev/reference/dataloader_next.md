# Get the next element of a dataloader iterator

Get the next element of a dataloader iterator

## Usage

``` r
dataloader_next(iter, completed = NULL)
```

## Arguments

- iter:

  a DataLoader iter created with
  [dataloader_make_iter](https://torch.mlverse.org/docs/dev/reference/dataloader_make_iter.md).

- completed:

  the returned value when the iterator is exhausted.
