BaseDatasetFetcher <- R6::R6Class(
  classname = "BaseDatasetFetcher",
  lock_objects = FALSE,
  public = list(
    initialize = function(dataset, auto_collation, collate_fn, drop_last) {
      self$dataset <- dataset
      self$auto_collation <- auto_collation
      self$collate_fn <- collate_fn
      self$drop_last <- drop_last
    },
    fetch = function(possibly_batched_index) {
      not_implemented_error()
    }
  )
)

IterableDatasetFetcher <- R6::R6Class(
  classname = "IterableDatasetFetcher",
  lock_objects = FALSE,
  inherit = BaseDatasetFetcher,
  public = list(
    initialize = function(dataset, auto_collation, collate_fn, drop_last) {
      super$initialize(dataset, auto_collation, collate_fn, drop_last)
      self$dataset_iter <- as_iterator(self$dataset)
    },
    fetch = function(possibly_batched_index) {
      if (self$auto_collation) {
        data <- vector(mode = "list", length = length(possibly_batched_index))
        for (i in seq_along(possibly_batched_index)) {
          d <- self$dataset_iter()

          if (coro::is_exhausted(d)) {
            break
          }

          data[[i]] <- d
        }
      } else {
        data <- self$dataset_iter()
      }
      self$collate_fn(data)
    }
  )
)

MapDatasetFetcher <- R6::R6Class(
  classname = "MapDatasetFetcher",
  lock_objects = FALSE,
  inherit = BaseDatasetFetcher,
  public = list(
    initialize = function(dataset, auto_collation, collate_fn, drop_last) {
      super$initialize(dataset, auto_collation, collate_fn, drop_last)
    },
    fetch = function(possibly_batched_index) {
      if (self$auto_collation) {
        data <- vector(mode = "list", length = length(possibly_batched_index))
        dataset <- self$dataset
        for (i in seq_along(data)) {
          data[[i]] <- dataset[possibly_batched_index[[i]]]
        }
      } else {
        data <- self$dataset[possibly_batched_index]
      }
      self$collate_fn(data)
    }
  )
)
