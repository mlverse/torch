#' @include optim.R
NULL

setdefault <- function(x, nm, default) {
  if (is.null(x[[nm]]))
    x[[nm]] <- default
  
  x
}

LRScheduler <- R6::R6Class(
  "LRScheduler",
  lock_objects = FALSE,
  public = list(
    initialize = function(optimizer, last_epoch = -1, verbose = FALSE) {
      
      if (!is_optimizer(optimizer))
        type_error("not an optimizer")
      
      self$optimizer <- optimizer
      
      if (last_epoch == -1) {
        
        optimizer$param_groups <- lapply(
          optimizer$param_groups,
          function(group) {
            setdefault(group, "initial_lr", group[["lr"]])
          }
        )
        
      } else {
        
        lapply(
          optimizer$param_groups,
          function(group) {
            if (is.null(group[["initial_lr"]]))
              value_error("param 'inital_lr' not is not specified.")
          }
        )
        
      }
      
      self$base_lrs <- lapply(optimizer$param_groups, 
                              function(group) group[["initial_lr"]])
      
      self$last_epoch <- last_epoch
      self$verbose <- verbose
      self$step()
      
    },
    
    state_dict = function() {
      dict <- as.list(self)
      dict <- dict[[-which(names(dict) == "optimizer")]]
      dict
    },
    
    load_state_dict = function(state_dict) {
      
      for (nm in names(state_dict)) {
        self[[nm]] <- state_dict[[nm]]
      }
      
    },
    
    get_last_lr = function() {
      self$.last_lr
    },
    
    get_lr = function() {
      not_implemented_error()
    },
    
    print_lr = function(is_verbose, group, lr, epoch = NULL) {
      
      if (is_verbose) {
        
        if (is.null(epoch)) {
          inform(sprintf("Adjusting learning rate of group %s to %.4f\n", group, lr))
        } else {
          inform(sprintf("Epoch %5d: adjusting learning rate of group %s to %.4f\n", epoch, group, lr))
        }
        
      }
      
    },
    
    step = function() {
      
      self$last_epoch <- self$last_epoch + 1
      values <- self$get_lr()
      
      for (i in seq_along(self$optimizer$param_groups)) {
        self$optimizer$param_groups[[i]]$lr <- values[i]
        self$print_lr(self$verbose, i, lr, epoch)
      }
      
      self$.last_lr <- sapply(self$optimizer$param_groups, function(x) x$lr)
    }
  )
)

#' Creates learning rate schedulers
#' 
#' @param classname optional name for the learning rate scheduler
#' @param inherit an optional learning rate scheduler to inherit from
#' @param ... named list of methods. You must implement the `get_lr()`
#'   method that doesn't take any argument and returns learning rates
#'   for each `param_group` in the optimizer.
#' @param parent_env passed to [R6::R6Class()].
#'
#' @export
lr_scheduler <- function(classname = NULL, inherit = LRScheduler, ..., 
                         parent_env = parent.frame()) {
  
  
  if (inherits(inherit, "lr_scheduler"))
    inherit <- attr(inherit, "scheduler")
  
  e <- new.env(parent = parent_env)
  e$inherit <- inherit
  
  classes <- c(classname, "lr_scheduler")
  
  Scheduler <- R6::R6Class(
    classname = classname,
    inherit = inherit,
    lock_objects = FALSE,
    public = list(
      .classes = classes,
      ...
    ),
    parent_env = e
  )
  
  init <- get_init(Scheduler)
  fun <- rlang::new_function(
    args = rlang::fn_fmls(init), 
    body = rlang::expr({
      instance <- Scheduler$new(!!!rlang::fn_fmls_syms(init))
      instance
    })
  )
  
  attr(fun, "class") <- c(classes, "lr_scheduler_generator")
  attr(fun, "scheduler") <- Scheduler
  fun
}

#' Sets the learning rate of each parameter group to the initial lr
#' times a given function. When last_epoch=-1, sets initial lr as lr.
#' 
#' @param optimizer (Optimizer): Wrapped optimizer.
#' @param lr_lambda (function or list): A function which computes a multiplicative
#'   factor given an integer parameter epoch, or a list of such
#'   functions, one for each group in optimizer.param_groups.
#' @param last_epoch (int): The index of last epoch. Default: -1.
#' @param verbose (bool): If `TRUE`, prints a message to stdout for
#'   each update. Default: `FALSE`.
#' 
#' @examples
#' # Assuming optimizer has two groups.
#' lambda1 <- function(epoch) epoch %/% 30
#' lambda2 <- function(epoch) 0.95^epoch
#' scheduler <- lr_lambda(optimizer, lr_lambda = list(lambda1, lambda2))
#' 
#' \dontrun{
#' for (epoch in 1:100) {
#'   train(...)
#'   validate(...)
#'   scheduler$step()
#' }
#' }
#' 
#' @export
lr_lambda <- lr_scheduler(
  "lr_lambda",
  initialize = function(optimizer, lr_lambda, last_epoch=-1, verbose=FALSE) {
    self$optimizer <- optimizer
    
    if (!is.list(lr_lambda)) {
      self$lr_lambdas <- lapply(
        seq_along(optimizer$param_groups), 
        function(i) lr_lambda
      )
    } else {
      
      if (length(lr_lambda) != length(optimizer$param_groups)) {
        i <- length(lr_lambda)
        j <- length(optimizer$param_groups)
        value_error("lr_lambda length ({i}) is different from the number of",
                    "optimizer$param_grpups ({j})")
      }
      
      self$lr_lambdas <- lr_lambda
    }
    
    super$initialize(optimizer, last_epoch, verbose)
  },
  
  get_lr = function() {
    
    lrs <- as.numeric(self$base_lrs)
    for (i in seq_along(lrs)) {
      lrs[i] <- lrs[i] * self$lr_lambdas[[i]](self$last_epoch)
    }
    
    lrs
  }
)
