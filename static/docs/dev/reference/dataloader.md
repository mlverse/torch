# Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

Data loader. Combines a dataset and a sampler, and provides single- or
multi-process iterators over the dataset.

## Usage

``` r
dataloader(
  dataset,
  batch_size = 1,
  shuffle = FALSE,
  sampler = NULL,
  batch_sampler = NULL,
  num_workers = 0,
  collate_fn = NULL,
  pin_memory = FALSE,
  drop_last = FALSE,
  timeout = -1,
  worker_init_fn = NULL,
  worker_globals = NULL,
  worker_packages = NULL
)
```

## Arguments

- dataset:

  (Dataset): dataset from which to load the data.

- batch_size:

  (int, optional): how many samples per batch to load (default: `1`).

- shuffle:

  (bool, optional): set to `TRUE` to have the data reshuffled at every
  epoch (default: `FALSE`).

- sampler:

  (Sampler, optional): defines the strategy to draw samples from the
  dataset. If specified, `shuffle` must be False. Custom samplers can be
  created with
  [`sampler()`](https://torch.mlverse.org/docs/dev/reference/sampler.md).

- batch_sampler:

  (Sampler, optional): like sampler, but returns a batch of indices at a
  time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and
  `drop_last`. Custom samplers can be created with
  [`sampler()`](https://torch.mlverse.org/docs/dev/reference/sampler.md).

- num_workers:

  (int, optional): how many subprocesses to use for data loading. 0
  means that the data will be loaded in the main process. (default: `0`)

- collate_fn:

  (callable, optional): merges a list of samples to form a mini-batch.

- pin_memory:

  (bool, optional): If `TRUE`, the data loader will copy tensors into
  CUDA pinned memory before returning them. If your data elements are a
  custom type, or your `collate_fn` returns a batch that is a custom
  type see the example below.

- drop_last:

  (bool, optional): set to `TRUE` to drop the last incomplete batch, if
  the dataset size is not divisible by the batch size. If `FALSE` and
  the size of dataset is not divisible by the batch size, then the last
  batch will be smaller. (default: `FALSE`)

- timeout:

  (numeric, optional): if positive, the timeout value for collecting a
  batch from workers. -1 means no timeout. (default: `-1`)

- worker_init_fn:

  (callable, optional): If not `NULL`, this will be called on each
  worker subprocess with the worker id (an int in `[1, num_workers]`) as
  input, after seeding and before data loading. (default: `NULL`)

- worker_globals:

  (list or character vector, optional) only used when `num_workers > 0`.
  If a character vector, then objects with those names are copied from
  the global environment to the workers. If a named list, then this list
  is copied and attached to the worker global environment. Notice that
  the objects are copied only once at the worker initialization.

- worker_packages:

  (character vector, optional) Only used if `num_workers > 0` optional
  character vector naming packages that should be loaded in each worker.

## Parallel data loading

When using `num_workers > 0` data loading will happen in parallel for
each worker. Note that batches are taken in parallel and not
observations.

The worker initialization process happens in the following order:

- `num_workers` R sessions are initialized.

Then in each worker we perform the following actions:

- the `torch` library is loaded.

- a random seed is set both using
  [`set.seed()`](https://rdrr.io/r/base/Random.html) and using
  `torch_manual_seed`.

- packages passed to the `worker_packages` argument are loaded.

- objects passed trough the `worker_globals` parameters are copied into
  the global environment.

- the `worker_init` function is ran with an `id` argument.

- the dataset fetcher is copied to the worker.

## See also

[`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md),
[`sampler()`](https://torch.mlverse.org/docs/dev/reference/sampler.md)
