# Creates a new Sampler

Samplers can be used with
[`dataloader()`](https://torch.mlverse.org/docs/dev/reference/dataloader.md)
when creating batches from a torch
[`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md).

## Usage

``` r
sampler(
  name = NULL,
  inherit = Sampler,
  ...,
  private = NULL,
  active = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- name:

  (optional) name of the sampler

- inherit:

  (optional) you can inherit from other samplers to re-use some methods.

- ...:

  Pass any number of fields or methods. You should at least define the
  `initialize` and `step` methods. See the examples section.

- private:

  (optional) a list of private methods for the sampler

- active:

  (optional) a list of active methods for the sampler.

- parent_env:

  used to capture the right environment to define the class. The default
  is fine for most situations.

## Details

A sampler must implement the `.iter` and `.length()` methods.

- `initialize` takes in a `data_source`. In general this is a
  [`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md).

- `.iter` returns a function that returns an integer vector or
  coro::exhausted(). For a sampler, the integer vector should have
  length 1 (the value is one data index). For a batch_sampler, the
  integer vector should have length equal to batch size (the values are
  indices in the batch).

- `.length` returns the maximum number of times that .iter() can be
  called, before it returns coro::exhausted(). For a sampler, this the
  number of samples. For a batch_sampler, this is the number of batches.
