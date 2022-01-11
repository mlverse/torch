#' @include optim.R
NULL

setdefault <- function(x, nm, default) {
  if (is.null(x[[nm]])) {
    x[[nm]] <- default
  }

  x
}

LRScheduler <- R6::R6Class(
  "LRScheduler",
  lock_objects = FALSE,
  public = list(
    initialize = function(optimizer, last_epoch = -1, verbose = FALSE) {
      if (!is_optimizer(optimizer)) {
        type_error("not an optimizer")
      }

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
            if (is.null(group[["initial_lr"]])) {
              value_error("param 'inital_lr' not is not specified.")
            }
          }
        )
      }

      self$base_lrs <- lapply(
        optimizer$param_groups,
        function(group) group[["initial_lr"]]
      )

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
    print_lr = function(is_verbose, group, lr) {
      if (is_verbose) {
        inform(sprintf("Adjusting learning rate of group %s to %.4f", group, lr))
      }
    },
    step = function() {
      self$last_epoch <- self$last_epoch + 1
      values <- self$get_lr()

      for (i in seq_along(self$optimizer$param_groups)) {
        self$optimizer$param_groups[[i]]$lr <- values[i]
        self$print_lr(self$verbose, i, self$optimizer$param_groups[[i]]$lr)
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
  if (inherits(inherit, "lr_scheduler")) {
    inherit <- attr(inherit, "scheduler")
  }

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
#' \dontrun{
#' scheduler <- lr_lambda(optimizer, lr_lambda = list(lambda1, lambda2))
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
  initialize = function(optimizer, lr_lambda, last_epoch = -1, verbose = FALSE) {
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
        value_error(
          "lr_lambda length ({i}) is different from the number of",
          "optimizer$param_grpups ({j})"
        )
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

#' Multiply the learning rate of each parameter group by the factor given
#' in the specified function. When last_epoch=-1, sets initial lr as lr.
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
#' \dontrun{
#' lmbda <- function(epoch) 0.95
#' scheduler <- lr_multiplicative(optimizer, lr_lambda = lmbda)
#' for (epoch in 1:100) {
#'   train(...)
#'   validate(...)
#'   scheduler$step()
#' }
#' }
#'
#' @export
lr_multiplicative <- lr_scheduler(
  "lr_multiplicative",
  initialize = function(optimizer, lr_lambda, last_epoch = -1, verbose = FALSE) {
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
        value_error(
          "lr_lambda length ({i}) is different from the number of",
          "optimizer$param_grpups ({j})"
        )
      }

      self$lr_lambdas <- lr_lambda
    }

    super$initialize(optimizer, last_epoch, verbose)
  },
  get_lr = function() {
    if (self$last_epoch > 0) {
      lrs <- numeric(length = length(self$optimizer$param_groups))
      for (i in seq_along(self$optimizer$param_groups)) {
        lrs[i] <- self$optimizer$param_groups[[i]]$lr * self$lr_lambdas[[i]](self$last_epoch)
      }
    } else {
      lrs <- as.numeric(self$base_lrs)
    }
    lrs
  }
)

#' Step learning rate decay
#'
#' Decays the learning rate of each parameter group by gamma every
#' step_size epochs. Notice that such decay can happen simultaneously with
#' other changes to the learning rate from outside this scheduler. When
#' last_epoch=-1, sets initial lr as lr.
#'
#'
#' @param optimizer (Optimizer): Wrapped optimizer.
#' @param step_size (int): Period of learning rate decay.
#' @param gamma (float): Multiplicative factor of learning rate decay.
#'   Default: 0.1.
#' @param last_epoch (int): The index of last epoch. Default: -1.
#'
#' @examples
#' \dontrun{
#' # Assuming optimizer uses lr = 0.05 for all groups
#' # lr = 0.05     if epoch < 30
#' # lr = 0.005    if 30 <= epoch < 60
#' # lr = 0.0005   if 60 <= epoch < 90
#' # ...
#' scheduler <- lr_step(optimizer, step_size = 30, gamma = 0.1)
#' for (epoch in 1:100) {
#'   train(...)
#'   validate(...)
#'   scheduler$step()
#' }
#' }
#'
#' @export
lr_step <- lr_scheduler(
  "lr_step",
  initialize = function(optimizer, step_size, gamma = 0.1, last_epoch = -1) {
    self$step_size <- step_size
    self$gamma <- gamma
    super$initialize(optimizer, last_epoch)
  },
  get_lr = function() {
    if ((self$last_epoch == 0) || (self$last_epoch %% self$step_size != 0)) {
      return(sapply(self$optimizer$param_groups, function(x) x$lr))
    }

    sapply(self$optimizer$param_groups, function(x) x$lr * self$gamma)
  }
)

#' Once cycle learning rate
#'
#' Sets the learning rate of each parameter group according to the
#' 1cycle learning rate policy. The 1cycle policy anneals the learning
#' rate from an initial learning rate to some maximum learning rate and then
#' from that maximum learning rate to some minimum learning rate much lower
#' than the initial learning rate.
#'
#' This policy was initially described in the paper
#' [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120).
#'
#' The 1cycle learning rate policy changes the learning rate after every batch.
#' `step` should be called after a batch has been used for training.
#' This scheduler is not chainable.
#'
#' Note also that the total number of steps in the cycle can be determined in one
#' of two ways (listed in order of precedence):
#'
#' - A value for total_steps is explicitly provided.
#' - A number of epochs (epochs) and a number of steps per epoch
#'   (steps_per_epoch) are provided.
#'
#' In this case, the number of total steps is inferred by
#' total_steps = epochs * steps_per_epoch
#'
#' You must either provide a value for total_steps or provide a value for both
#' epochs and steps_per_epoch.
#'
#'
#' @param optimizer (Optimizer): Wrapped optimizer.
#' @param max_lr (float or list): Upper learning rate boundaries in the cycle
#'   for each parameter group.
#' @param total_steps (int): The total number of steps in the cycle. Note that
#'   if a value is not provided here, then it must be inferred by providing
#'   a value for epochs and steps_per_epoch.
#'   Default: NULL
#' @param epochs (int): The number of epochs to train for. This is used along
#'   with steps_per_epoch in order to infer the total number of steps in the cycle
#'   if a value for total_steps is not provided.
#'   Default: NULL
#' @param steps_per_epoch (int): The number of steps per epoch to train for. This is
#'   used along with epochs in order to infer the total number of steps in the
#'   cycle if a value for total_steps is not provided.
#'   Default: NULL
#' @param pct_start (float): The percentage of the cycle (in number of steps) spent
#'   increasing the learning rate.
#'   Default: 0.3
#' @param anneal_strategy (str): {'cos', 'linear'}
#'   Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
#'   linear annealing.
#'   Default: 'cos'
#' @param cycle_momentum (bool): If `TRUE`, momentum is cycled inversely
#'   to learning rate between 'base_momentum' and 'max_momentum'.
#'   Default: TRUE
#' @param base_momentum (float or list): Lower momentum boundaries in the cycle
#'   for each parameter group. Note that momentum is cycled inversely
#'   to learning rate; at the peak of a cycle, momentum is
#'   'base_momentum' and learning rate is 'max_lr'.
#'   Default: 0.85
#' @param max_momentum (float or list): Upper momentum boundaries in the cycle
#'   for each parameter group. Functionally,
#'   it defines the cycle amplitude (max_momentum - base_momentum).
#'   Note that momentum is cycled inversely
#'   to learning rate; at the start of a cycle, momentum is 'max_momentum'
#'   and learning rate is 'base_lr'
#'   Default: 0.95
#' @param div_factor (float): Determines the initial learning rate via
#'   initial_lr = max_lr/div_factor
#'   Default: 25
#' @param final_div_factor (float): Determines the minimum learning rate via
#'   min_lr = initial_lr/final_div_factor
#'   Default: 1e4
#' @param last_epoch (int): The index of the last batch. This parameter is used when
#'   resuming a training job. Since `step()` should be invoked after each
#'   batch instead of after each epoch, this number represents the total
#'   number of *batches* computed, not the total number of epochs computed.
#'   When last_epoch=-1, the schedule is started from the beginning.
#'   Default: -1
#' @param verbose (bool): If `TRUE`, prints a message to stdout for
#'   each update. Default: `FALSE`.
#'
#' @examples
#' \dontrun{
#' data_loader <- dataloader(...)
#' optimizer <- optim_sgd(model$parameters, lr = 0.1, momentum = 0.9)
#' scheduler <- lr_one_cycle(optimizer,
#'   max_lr = 0.01, steps_per_epoch = length(data_loader),
#'   epochs = 10
#' )
#'
#' for (i in 1:epochs) {
#'   coro::loop(for (batch in data_loader) {
#'     train_batch(...)
#'     scheduler$step()
#'   })
#' }
#' }
#'
#' @export
lr_one_cycle <- lr_scheduler(
  "lr_one_cycle",
  initialize = function(optimizer,
                        max_lr,
                        total_steps = NULL,
                        epochs = NULL,
                        steps_per_epoch = NULL,
                        pct_start = 0.3,
                        anneal_strategy = "cos",
                        cycle_momentum = TRUE,
                        base_momentum = 0.85,
                        max_momentum = 0.95,
                        div_factor = 25.,
                        final_div_factor = 1e4,
                        last_epoch = -1,
                        verbose = FALSE) {
    self$optimizer <- optimizer

    # Validate total_steps
    if (is.null(total_steps) && is.null(epochs) && is.null(steps_per_epoch)) {
      value_error("You must define either total_steps OR (epochs AND steps_per_epoch)")
    } else if (!is.null(total_steps)) {
      if (!is.numeric(total_steps) || total_steps <= 0) {
        value_error("Expected positive integer total_steps, but got {total_steps}")
      }

      self$total_steps <- total_steps
    } else {
      if (!is.numeric(epochs) || epochs <= 0) {
        value_error("Expected positive integer epochs, but got {epochs}")
      }

      if (!is.numeric(steps_per_epoch) || steps_per_epoch <= 0) {
        value_error("Expected positive integer steps_per_epoch, but got {steps_per_epoch}")
      }

      self$total_steps <- epochs * steps_per_epoch
    }

    self$step_size_up <- (pct_start * self$total_steps) - 1
    self$step_size_down <- (self$total_steps - self$step_size_up) - 1

    # Validate pct_start
    if (!is.numeric(pct_start) || pct_start < 0 || pct_start > 1) {
      value_error("Expected float between 0 and 1 pct_start, but got {pct_start}")
    }

    # Validate anneal_strategy
    if (!anneal_strategy %in% c("cos", "linear")) {
      value_error("anneal_strategy must by one of 'cos' or 'linear', instead got {anneal_strategy}")
    } else if (anneal_strategy == "cos") {
      self$anneal_func <- self$.annealing_cos
    } else if (anneal_strategy == "linear") {
      self.anneal_func <- self$.annealing_linear
    }

    # Initialize learning rate variables
    max_lrs <- self$.format_param("max_lr", self$optimizer, max_lr)
    if (last_epoch == -1) {
      for (i in seq_along(self$optimizer$param_groups)) {
        self$optimizer$param_groups[[i]][["initial_lr"]] <- max_lrs[[i]] / div_factor
        self$optimizer$param_groups[[i]][["max_lr"]] <- max_lrs[[i]]
        self$optimizer$param_groups[[i]][["min_lr"]] <- self$optimizer$param_groups[[i]][["initial_lr"]] /
          final_div_factor
      }
    }

    # Initialize momentum variables
    self$cycle_momentum <- cycle_momentum

    if (self$cycle_momentum) {
      if ((!"momentum" %in% names(self$optimizer$defaults)) &&
        !("betas" %in% names(self$optimizer$defaults))) {
        value_error("optimizer must support momentum with `cycle momentum` enabled")
      }

      self$use_beta1 <- "betas" %in% names(self$optimizer$defaults)
      max_momentums <- self$.format_param("max_momentum", optimizer, max_momentum)
      base_momentums <- self$.format_param("base_momentum", optimizer, base_momentum)

      if (last_epoch == -1) {
        for (i in seq_along(self$optimizer$param_groups)) {
          if (self$use_beta1) {
            beta2 <- self$optimizer$param_groups[[i]]$betas[[2]]
            self$optimizer$param_groups[[i]]$betas <- c(max_momentum[[i]], beta2)
          } else {
            self$optimizer$param_groups[[i]]$momentum <- max_momentum[[i]]
          }

          self$optimizer$param_groups[[i]]$max_momentum <- max_momentum[[i]]
          self$optimizer$param_groups[[i]]$base_momentum <- base_momentums[[i]]
        }
      }
    }

    super$initialize(optimizer, last_epoch, verbose)
  },
  .format_param = function(name, optimizer, param) {
    if (is.list(param) || length(param) > 1) {
      if (length(param) != length(optimizer$param_groups)) {
        value_error(
          "expected {length(optimizer$param_groups)} values for {name}",
          "but got {length(param)}"
        )
      }

      return(param)
    } else {
      return(lapply(seq_along(optimizer$param_groups), function(x) param))
    }
  },
  .annealing_cos = function(start, end, pct) {
    cos_out <- cos(pi * pct) + 1
    end + (start - end) / 2.0 * cos_out
  },
  .annealing_linear = function(start, end, pct) {
    (end - start) * pct + start
  },
  get_lr = function() {
    lrs <- list()
    step_num <- self$last_epoch

    if (step_num > self$total_steps) {
      value_error(
        "Tried to step {step_num+1} times. The specified number of total steps is {self$total_steps}"
      )
    }

    for (i in seq_along(self$optimizer$param_groups)) {
      if (step_num <= self$step_size_up) {
        computed_lr <- self$anneal_func(
          self$optimizer$param_groups[[i]][["initial_lr"]],
          self$optimizer$param_groups[[i]][["max_lr"]],
          step_num / self$step_size_up
        )

        if (self$cycle_momentum) {
          computed_momentum <- self$anneal_func(
            self$optimizer$param_groups[[i]][["max_momentum"]],
            self$optimizer$param_groups[[i]][["base_momentum"]],
            step_num / self$step_size_up
          )
        }
      } else {
        down_step_num <- step_num - self$step_size_up
        computed_lr <- self$anneal_func(
          self$optimizer$param_groups[[i]][["max_lr"]],
          self$optimizer$param_groups[[i]][["min_lr"]],
          down_step_num / self$step_size_down
        )

        if (self$cycle_momentum) {
          computed_momentum <- self$anneal_func(
            self$optimizer$param_groups[[i]][["base_momentum"]],
            self$optimizer$param_groups[[i]][["max_momentum"]],
            down_step_num / self$step_size_down
          )
        }
      }

      lrs[[i]] <- computed_lr
      if (self$cycle_momentum) {
        if (self$use_beta1) {
          beta2 <- self$optimizer$param_groups[[i]][["betas"]][[2]]
          self$optimizer$param_groups[[i]][["betas"]] <- c(computed_momentum, beta2)
        } else {
          self$optimizer$param_groups[[i]][["momentum"]] <- computed_momentum
        }
      }
    }

    as.numeric(lrs)
  }
)
