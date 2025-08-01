#' @export
length.dataloader <- function(x) {
  x$.length()
}

#' Creates an iterator from a DataLoader
#'
#' @param dataloader a dataloader object.
#'
#' @export
dataloader_make_iter <- function(dataloader) {
  dataloader$.iter()
}

#' Checks if the object is a dataloader
#'
#' @param x object to check
#'
#' @export
is_dataloader <- function(x) {
  inherits(x, "dataloader")
}

#' Get the next element of a dataloader iterator
#'
#' @param iter a DataLoader iter created with [dataloader_make_iter].
#' @param completed the returned value when the iterator is exhausted.
#'
#' @export
dataloader_next <- function(iter, completed = NULL) {
  res <- iter$.next()
  if (is_exhausted(res)) {
    completed
  } else {
    res
  }
}

#' Data loader. Combines a dataset and a sampler, and provides
#' single- or multi-process iterators over the dataset.
#'
#' @param dataset (Dataset): dataset from which to load the data.
#' @param batch_size (int, optional): how many samples per batch to load
#'   (default: `1`).
#' @param shuffle (bool, optional): set to `TRUE` to have the data reshuffled
#'   at every epoch (default: `FALSE`).
#' @param sampler (Sampler, optional): defines the strategy to draw samples from
#'   the dataset. If specified, `shuffle` must be False. Custom samplers can be
#'   created with [sampler()].
#' @param batch_sampler (Sampler, optional): like sampler, but returns a batch of
#'   indices at a time. Mutually exclusive with `batch_size`,
#'   `shuffle`, `sampler`, and `drop_last`. Custom samplers can be created with
#'   [sampler()].
#' @param num_workers (int, optional): how many subprocesses to use for data
#'   loading. 0 means that the data will be loaded in the main process.
#'   (default: `0`)
#' @param collate_fn (callable, optional): merges a list of samples to form a mini-batch.
#' @param pin_memory (bool, optional): If `TRUE`, the data loader will copy tensors
#'   into CUDA pinned memory before returning them.  If your data elements
#'   are a custom type, or your `collate_fn` returns a batch that is a custom type
#'   see the example below.
#' @param drop_last (bool, optional): set to `TRUE` to drop the last incomplete batch,
#'   if the dataset size is not divisible by the batch size. If `FALSE` and
#'   the size of dataset is not divisible by the batch size, then the last batch
#'   will be smaller. (default: `FALSE`)
#' @param timeout (numeric, optional): if positive, the timeout value for collecting a batch
#'   from workers. -1 means no timeout. (default: `-1`)
#' @param worker_init_fn (callable, optional): If not `NULL`, this will be called on each
#'   worker subprocess with the worker id (an int in `[1, num_workers]`) as
#'   input, after seeding and before data loading. (default: `NULL`)
#' @param worker_globals (list or character vector, optional) only used when
#'   `num_workers > 0`. If a character vector, then objects with those names are
#'   copied from the global environment to the workers. If a named list, then
#'   this list is copied and attached to the worker global environment. Notice
#'   that the objects are copied only once at the worker initialization.
#' @param worker_packages (character vector, optional) Only used if `num_workers > 0`
#'   optional character vector naming packages that should be loaded in
#'   each worker.
#'
#' @section Parallel data loading:
#'
#' When using `num_workers > 0` data loading will happen in parallel for each
#' worker. Note that batches are taken in parallel and not observations.
#'
#' The worker initialization  process happens in the following order:
#'
#' - `num_workers` R sessions are initialized.
#'
#' Then in each worker we perform the following actions:
#'
#' - the `torch` library is loaded.
#' - a random seed is set both using `set.seed()` and using `torch_manual_seed`.
#' - packages passed to the `worker_packages` argument are loaded.
#' - objects passed trough the `worker_globals` parameters are copied into the
#'   global environment.
#' - the `worker_init` function is ran with an `id` argument.
#' - the dataset fetcher is copied to the worker.
#' 
#' @seealso [dataset()], [sampler()]
#'
#'
#' @export
dataloader <- function(dataset, batch_size = 1, shuffle = FALSE,
                       sampler = NULL,
                       batch_sampler = NULL, num_workers = 0, collate_fn = NULL,
                       pin_memory = FALSE, drop_last = FALSE, timeout = -1,
                       worker_init_fn = NULL, worker_globals = NULL, worker_packages = NULL) {
  multiprocessing_context <- NULL
  generator <- NULL

  # find worker globals before stepping into the class env.
  if (is.character(worker_globals)) {
    worker_globals <- rlang::env_get_list(
      env = rlang::caller_env(),
      nms = worker_globals, inherit = TRUE,
      default = structure("", class = "notfound")
    )

    if (any(b <- sapply(worker_globals, inherits, "notfound"))) {
      runtime_error(
        "Could not find an object with name '{names(worker_globals)[b]}'."
      )
    }
  }

  DataLoader$new(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
    collate_fn, pin_memory, drop_last, timeout, worker_init_fn,
    multiprocessing_context, generator,
    worker_globals = worker_globals,
    worker_packages = worker_packages
  )
}

DataLoader <- R6::R6Class(
  classname = "dataloader",
  lock_objects = FALSE,
  public = list(
    initialize = function(dataset, batch_size = 1, shuffle = FALSE, sampler = NULL,
                          batch_sampler = NULL, num_workers = 0, collate_fn = NULL,
                          pin_memory = FALSE, drop_last = FALSE, timeout = 0,
                          worker_init_fn = NULL, multiprocessing_context = NULL,
                          generator = NULL, worker_globals = NULL,
                          worker_packages = NULL) {
      self$dataset <- dataset
      self$num_workers <- num_workers
      self$pin_memory <- pin_memory
      self$timeout <- timeout
      self$worker_init_fn <- worker_init_fn
      self$multiprocessing_context <- multiprocessing_context
      self$worker_globals <- worker_globals
      self$worker_packages <- worker_packages

      if (is_map_dataset(dataset)) {
        self$.dataset_kind <- "map"
      } else if (is_iterable_dataset(dataset)) {
        self$.dataset_kind <- "iterable"
      } else {
        cli_abort("Unknown dataset type with class {.cls {class(dataset)}}")
      }

      if (is.null(sampler)) {
        if (self$.dataset_kind == "iterable") {
          sampler <- InfiniteSampler()
        } else {
          if (shuffle) {
            sampler <- RandomSampler(dataset, generator = generator)
          } else {
            sampler <- SequentialSampler(dataset)
          }
        }
      }

      self$.has_getbatch <- !is.null(dataset$.getbatch)

      if (!is.null(batch_size) && is.null(batch_sampler)) {
        batch_sampler <- BatchSampler(sampler, batch_size, drop_last)
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

      # check that the dataset don't contain tensors in it and raise warnings.
      if (self$num_workers > 0) {
        walk_r6_instance_fields(self$dataset, warn_tensor)
      }
    },
    .iter = function() {
      if (self$.dataset_kind == "map") {
        if (self$num_workers == 0) {
          return(SingleProcessDataLoaderIter$new(self))
        }

        MultiProcessingDataLoaderIter$new(self)
      } else if (self$.dataset_kind == "iterable") {
        if (self$num_workers == 0) {
          return(SingleProcessDataLoaderIter$new(self))
        }
        cli_abort("Multi-process dataloader not implemented yet for Iterable datasets.")
      } else {
        not_implemented_error()
      }
    },
    .length = function() {
      if (self$.dataset_kind == "iterable") {
        l <- length(self$dataset)
        
        if (is.na(l)) return(l)
        
        if (self$drop_last) {
          return(l %/% self$batch_size)
        } else {
          return(as.integer(ceiling(l / self$batch_size)))
        } 
      } else {
        length(self$.index_sampler)
      }
    }
  ),
  active = list(
    .auto_collation = function() {
      !is.null(self$batch_sampler) && !self$.has_getbatch
    },
    .index_sampler = function() {
      if (self$.auto_collation || self$.has_getbatch) {
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
      self$.pin_memory <- loader$pin_memory # TODO && torch.cuda.is_available()
      self$.timeout <- loader$timeout
      self$.collate_fn <- loader$collate_fn
      self$.worker_init_fn <- loader$worker_init_fn
      self$.sampler_iter <- self$.index_sampler$.iter()
      self$.worker_globals <- loader$worker_globals
      self$.worker_packages <- loader$worker_packages
      # self$.base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
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

#' @importFrom coro is_exhausted
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
      } else if (self$.dataset_kind == "iterable") {
        self$.dataset_fetcher <- IterableDatasetFetcher$new(
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

      if (is_exhausted(index)) {
        return(coro::exhausted())
      }

      # data can be exhausted in iterable datasets
      data <- self$.dataset_fetcher$fetch(index)
      
      if (self$.pin_memory) {
        # TODO
      }
      data
    }
  )
)

#' @importFrom callr r_session
MultiProcessingDataLoaderIter <- R6::R6Class(
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

      # we call gc here because on Windows it seems that R is not aware of the
      # memory allocated by the subsessions, so it doesn't automatically call GC.
      # We can endup with too many worker sessions if calling dataloader in a
      # loop. See https://github.com/mlverse/torch/issues/622 for more info.
      gc()

      # initialize all the worker sections
      # using callr
      private$workers <- list()
      for (i in seq_len(self$.num_workers)) {
        private$workers[[i]] <- r_session$new()
      }

      worker_config <- function(id, num_workers, seed, init_fn, globals,
                                packages, socket_port = NULL) {
        library(torch)
        .worker_info <<- list(
          id = id,
          workers = num_workers,
          seed = seed
        )

        torch::torch_set_num_threads(1)
        set.seed(seed)
        torch::torch_manual_seed(seed)

        # load requested packages
        lapply(packages, function(x) library(x, character.only = TRUE))

        # copy globals to global env
        if (!is.null(globals)) {
          list2env(globals, envir = .GlobalEnv)
        }

        if (!is.null(init_fn)) {
          init_fn(id)
        }
        
        .socket_con <<- NULL
        if (!is.null(socket_port)) {
          # We need to wait for the main process to start the server, so here we
          # retry a few times until the conection works.
          for(i in 1:20) {
            tr <- try({.socket_con <<- socketConnection(
              port = socket_port, 
              blocking = TRUE, 
              open = "a+b"
            )}, silent = TRUE) 
            
            if (!inherits(tr, "try-error")) break
            Sys.sleep(0.5)
            
            if (i == 20) {
              runtime_error("Could not create a connection with the main process.")
            }
          }
        }
        
      }

      fetcher <- self$.dataset_fetcher$fetch
      # initialize the workers!
      for (i in seq_len(self$.num_workers)) {
        worker <- private$workers[[i]]

        # Creates initial worker configuration
        worker$session$call(
          worker_config,
          args = list(
            id = i,
            num_workers = self$.num_workers,
            seed = sample.int(1e6, 1),
            init_fn = self$.worker_init_fn,
            globals = self$.worker_globals,
            packages = self$.worker_packages,
            socket_port = worker$port
          )
        )
        
        # if using socket connections we need to start them now on the main
        # process. This will also ensure that the worker config function can
        # finish correctly.
        if (worker$using_socket_con) {
          worker$setup_socket_con()
        }
        
        worker$session$poll_process(-1)
        status <- worker$session$read() ## just so the worker is marked as idle
        if (status$code != 200) {
          runtime_error("Failed starting the worker.")
        }

        # move fetcher to each session
        worker$session$run(
          function(fetcher) {
            fetcher <<- fetcher
          },
          list(fetcher = fetcher)
        )
      }
    },
    .add_task = function() {
      index <- self$.next_index()

      # find first idle worker
      for (worker_id in seq_len(self$.num_workers)) {
        worker <- private$workers[[worker_id]]
        if (worker$session$get_state() == "idle") {
          break
        }
      }

      # send task to the worker
      if (coro::is_exhausted(index)) {
        worker$session$call(function() {
          torch:::to_exportable_tensor(coro::exhausted(), .socket_con)
        })
      } else {
        worker$session$call(function(index) {
          torch:::to_exportable_tensor(fetcher(index), .socket_con)
        }, list(index = index))
      }

      # adds a reference of that worker to the task list
      private$tasks[[length(private$tasks) + 1]] <- worker
    },
    .pop_task = function() {

      # get task and remove from the list
      task <- private$tasks[[1]]
      private$tasks <- private$tasks[-1]

      if (!task$using_socket_con) {
        # wait for the process to be ready
        p <- task$session$poll_process(timeout = self$.timeout)
        if (p == "timeout") {
          runtime_error("dataloader worker timed out.")
        }
        # read results
        result <- task$session$read() 
        
        # Raise error that might have happened in the subprocess.
        if (!is.null(result$error)) {
          if (packageVersion("callr") >= "3.7.1") {
            rlang::abort(
              "Error when getting dataset item.", 
              parent = result$error, 
              class = "runtime_error"
            )
          } else {
            runtime_error(result$error$message)  
          }
        }
        
        data <- result$result
        from_exportable_tensor(data)
      } else {
        
        socket_select_timeout <- 0.1
        time_passed <- 0
        while(!socketSelect(list(task$con), timeout = socket_select_timeout)) {
          result <- task$session$read()
          
          # no result means that the process is still runing, ie, no error
          # happened.
          if (!is.null(result)) {
            # the $error field is not NULL means that an error hapened in the
            # worker.
            if (!is.null(result$error)) {
              runtime_error(result$error$message)
            } else {
              # if no error happened and we got a result, it means that the socket
              # conection is now ready for read. so we avoid reaching timeout.
              next
            }
          }
          # timeout for the dataloader is in milliseconds.
          time_passed <- time_passed + socket_select_timeout
          if (self$.timeout > 0 && (time_passed*1000) > self$.timeout) {
            runtime_error("dataloader worker timed out.")
          }
        }
        
        out <- from_exportable_tensor(task$con)
        # we have to read so the worker becomes idle
        result <- task$session$read() 
        out
      }
    },
    .next_data = function() {
      workers <- self$.num_workers

      # the first time we call .next_data
      # we want to start num_worker tasks
      if (self$.num_yielded == 0) {
        start_n <- workers
      } else {
        start_n <- 1
      }

      for (i in seq_len(start_n)) {
        self$.add_task()
      }

      data <- self$.pop_task()
      if (self$.pin_memory) {
        # TODO
      }
      data
    }
  ),
  private = list(
    tasks = list(),
    finalize = function() {
      lapply(private$workers, function(x) {
        x$close_socket_con()
      })
    }
  )
)

#' Re-exporting the as_iterator function.
#' @importFrom coro as_iterator
#' @export
coro::as_iterator

#' Re-exporting the loop function.
#' @importFrom coro loop
#' @export
coro::loop

#' Re-exporting the iterate function.
#' @importFrom coro yield
#' @export
coro::yield

#' @importFrom coro as_iterator
#' @export
#' @method as_iterator dataloader
as_iterator.dataloader <- function(x) {
  iter <- dataloader_make_iter(x)
  coro::as_iterator(function() {
    dataloader_next(iter, coro::exhausted())
  })
}

# takes a tensor and saves it's state in a field so we can
# reconstruct it after transfering via futures
to_exportable_tensor <- function(x, con) {
  if (is.null(con)) {
    return(tensor_to_raw_vector(x))
  }
  torch_save(x, path = con)
  TRUE
}

from_exportable_tensor <- function(x) {
  if (!inherits(x, "connection")) {
    con <- rawConnection(x)
    on.exit({close(con)})
  } else {
    con <- x
  }
  torch_load(con)
}

walk_fields <- function(env, nms, func) {
  for (nm in nms) {
    func(env[[nm]], nm)
  }
}

walk_r6_instance_fields <- function(instance, func) {

  # find active fields - those should not be checked if they are tensors or
  # not.
  active <- instance$.__enclos_env__$.__active__

  nms <- rlang::env_names(instance)
  nms <- nms[!nms %in% active]

  public <- walk_fields(instance, nms, func)
  p_env <- instance$.__enclos_env__$private

  if (!is.null(p_env)) {
    private <- walk_fields(p_env, rlang::env_names(p_env), func)
  }
}

warn_tensor <- function(x, nm) {
  if (is_torch_tensor(x)) {
    rlang::warn(c(
      "Datasets used with parallel dataloader (num_workers > 0) shouldn't have fields containing tensors as they can't be correctly passed to the wroker subprocesses.",
      glue::glue("A field named '{nm}' exists.")
    ))
  } else if (is.list(x)) {
    imap(x, ~ warn_tensor(.x, paste0(nm, "$", .y)))
  }
}


r_session <- R6::R6Class(
  "r_session",
  public = list(
    port = NULL,
    con = NULL,
    session = NULL,
    using_socket_con = FALSE,
    initialize = function() {
      if (use_socket_con()) {
        self$port <- parallelly::freePort()  
        self$using_socket_con <- TRUE
      }
      self$session <- callr::r_session$new()
    },
    setup_socket_con = function() {
      self$con <- socketConnection(
        port = self$port, server = TRUE, 
        blocking = TRUE, open = "a+b")
    },
    close_socket_con = function() {
      if (self$using_socket_con) {
        try(close(self$con), silent = TRUE)
        self$con <- NULL
      }
    }
  )
)

use_socket_con <- function() {
  getOption("torch.dataloader_use_socket_con", FALSE)
}
