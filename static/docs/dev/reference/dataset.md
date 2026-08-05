# Helper function to create an function that generates R6 instances of class `dataset`

All datasets that represent a map from keys to data samples should
subclass this class. All subclasses should overwrite the `.getitem()`
method, which supports fetching a data sample for a given key.
Subclasses could also optionally overwrite `.length()`, which is
expected to return the size of the dataset (e.g. number of samples) used
by many sampler implementations and the default options of
[`dataloader()`](https://torch.mlverse.org/docs/dev/reference/dataloader.md).

## Usage

``` r
dataset(
  name = NULL,
  inherit = Dataset,
  ...,
  private = NULL,
  active = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- name:

  a name for the dataset. It it's also used as the class for it.

- inherit:

  you can optionally inherit from a dataset when creating a new dataset.

- ...:

  public methods for the dataset class

- private:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).

- active:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).

- parent_env:

  An environment to use as the parent of newly-created objects.

## Value

The output is a function `f` with class `dataset_generator`. Calling
`f()` creates a new instance of the R6 class `dataset`. The R6 class is
stored in the enclosing environment of `f` and can also be accessed
through `f`s attribute `Dataset`.

## Note

[`dataloader()`](https://torch.mlverse.org/docs/dev/reference/dataloader.md)
by default constructs a index sampler that yields integral indices. To
make it work with a map-style dataset with non-integral indices/keys, a
custom sampler must be provided.

## Get a batch of observations

By default datasets are iterated by returning each observation/item
individually. Often it's possible to have an optimized implementation to
take a batch of observations (eg, subsetting a tensor by multiple
indexes at once is faster than subsetting once for each index), in this
case you can implement a `.getbatch` method that will be used instead of
`.getitem` when getting a batch of observations within the dataloader.
`.getbatch` must work for batches of size larger or equal to 1 and care
must be taken so it doesn't drop the batch dimension when it's queried
with a length 1 batch index - for instance by using `drop=FALSE`.
`.getitem()` is expected to not include the batch dimension as it's
added by the datalaoder. For more on this see the the
[`vignette("loading-data")`](https://torch.mlverse.org/docs/dev/articles/loading-data.md).
