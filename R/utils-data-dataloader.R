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
      
      
      if (inherits(datset, "Dataset"))
        self$dataset_kind_ <- "map"
      
      
      if (is.null(sampler)) {
        
        if (self$dataset_kind_ == "iterable") {
          #TODO
        } else {
          
          if (shuffle) {
            sampler <- RandomSampler$new(dataset, generator)
          } else {
            sampler <- SequentialSampler$new(dataset, generator)
          }
          
        }
        
      }
      
      if (!is.null(batch_size) && !is.null(batch_sampler)) {
        batch_sampler <- BatchSampler$new(sampler, batch_size, drop_last)
      }
      
      self$batch_size <- batch_size
      self$drop_last <- drop_last
      self$sampler <- sampler
      self$batch_sampler <- batch_sampler
      self$generator <- generator
      
      if (is.null(collate_fn)) {
        
        if (self$auto_collation_) {
          collate_fn <- utils_data_default_collate
        } else {
          # TODO
        }
        
      }
      
      self$collate_fn <- collate_fn
      
    },
    iter = function() {
      
      if (self$dataset_kind_ == "map") {
        
      } else {
        not_implemented_error()
      }
        
      
    }
  ),
  active = list(
    auto_collation_ = function() {
      !is.null(self$batch_sampler)
    }
  )
)