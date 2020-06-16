BaseDatasetFetcher <- R6::R6Class(
  classname = "BaseDatasetFetcher",
  lock_objects = FALSE,
  public = list(
    initialize = function(dataset, auto_collation, collate_fn, drop_last) {
      self$dataset <- dataset
      self$auto_collation <-  auto_collation
      self$collate_fn <- collate_fn
      self$drop_last <- drop_last
    },
    fetch = function(possibly_batched_index) {
      not_implemented_error()
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
        data <- lapply(possibly_batched_index, function(idx) self$dataset[idx])
      } else {
        data <- self$dataset[possibly_batched_index]
      }
      self$collate_fn(data)
    }
  )
)
