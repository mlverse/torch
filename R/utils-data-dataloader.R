#' @export
length.utils_data_loader <- function(x) {
  x$.lenght()
}

DataLoader <- R6::R6Class(
  classname = "utils_data_loader",
  lock_objects = FALSE,
  public = list(
    initialize = function(dataset, batch_size=1, shuffle=FALSE, sampler=NULL,
                          batch_sampler=NULL, num_workers=0, collate_fn=NULL,
                          pin_memory=FALSE, drop_last=FALSE, timeout=0,
                          worker_init_fn=NULL, multiprocessing_context=NULL,
                          generator=NULL) {
      
      
      self$dataset <- dataset
      self$num_workers <- num_workers
      self$pin_memory <- pin_memory
      self$timeout <- timeout
      self$worker_init_fn <- worker_init_fn
      self$multiprocessing_context <- multiprocessing_context
      
      if (is_map_dataset(dataset))
        self$.dataset_kind <- "map"
      
      if (is.null(sampler)) {
        
        if (self$.dataset_kind == "iterable") {
          #TODO
        } else {
          
          if (shuffle) {
            sampler <- RandomSampler$new(dataset, generator = generator)
          } else {
            sampler <- SequentialSampler$new(dataset)
          }
          
        }
        
      }
      
      if (!is.null(batch_size) && is.null(batch_sampler)) {
        batch_sampler <- BatchSampler$new(sampler, batch_size, drop_last)
      }
      
      self$batch_size <- batch_size
      self$drop_last <- drop_last
      self$sampler <- sampler
      self$batch_sampler <- batch_sampler
      self$generator <- generator
      
      if (is.null(collate_fn)) {
        
        if (self$.auto_collation) {
          collate_fn <- utils_data_default_collate
        } else {
          collate_fn <- utils_data_default_convert
        }
        
      }
      
      self$collate_fn <- collate_fn
      
    },
    .iter = function() {
      
      if (self$.dataset_kind == "map") {
        return(SingleProcessDataLoaderIter$new(self))
      } else {
        not_implemented_error()
      }
        
    },
    .lenght = function() {
      if (self$.dataset_kind == "iterable") {
        not_implemented_error()
      } else {
        length(self$.index_sampler)
      }
    }
  ),
  active = list(
    .auto_collation = function() {
      !is.null(self$batch_sampler)
    },
    .index_sampler = function() {
      if (self$.auto_collation) {
        return(self$batch_sampler)
      } else {
        return(self$sampler)
      }
    }
  )
)

BaseDataLoaderIter <- R6::R6Class(
  classname = "BaseDataLoaderIter",
  public = list(
    initialize = function(loader) {
      self$.dataset <- loader$dataset
      self$.dataset_kind <- loader$.dataset_kind
      self$.IterableDataset_len_called <- loader$.IterableDataset_len_called
      self$.auto_collation <- loader$.auto_collation
      self$.drop_last <- loader$drop_last
      self$.index_sampler <- loader$.index_sampler
      self$.num_workers <- loader$num_workers
      self$.pin_memory <- loader$pin_memory #TODO && torch.cuda.is_available()
      self$.timeout <- loader$timeout
      self$.collate_fn <- loader$collate_fn
      self$.sampler_iter <- self$.index_sampler$.iter()
      #self$.base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
      self$.num_yielded <- 0
    },
    .iter = function() {
      self
    },
    .next_index = function() {
      self$.sampler_iter()
    },
    .next_data = function() {
      not_implemented_error()
    },
    .next = function() {
      data <- self$.next_data()
      self$.num_yielded <- self$.num_yielded + 1
      data
    },
    .length = function() {
      length(self$.index_sampler)
    }
  )
)

SingleProcessDataLoaderIter <- R6::R6Class(
  classname = "SingleProcessDataLoaderIter",
  inherit = BaseDataLoaderIter,
  lock_objects = FALSE,
  public = list(
    initialize = function(loader) {
      
      super$initialize(loader)
      
      if (self$.dataset_kind == "map") {
        self$.dataset_fetcher <- MapDatasetFetcher$new(
          self$.dataset, 
          self$.auto_collation, 
          self$.collate_fn, 
          self$.drop_last
        ) 
      } else {
        not_implemented_error()
      }
      
    },
    .next_data = function() {
      index <- self$.next_index()
      data <- self$.dataset_fetcher$fetch(index)
      if (self$.pin_memory) {
        # TODO
      }
      data
    }
  )
)