#' Abstract base class for constraints.
#'
#' A constraint object represents a region over which a variable is valid,
#' e.g. within which a variable can be optimized.
#'
Constraint <- R6::R6Class(
  "torch_Constraint",
  lock_objects = FALSE,
  public = list(

    #' @description
    #' Returns a byte tensor of `sample_shape + batch_shape` indicating
    #' whether each event in value satisfies this constraint.
    #'
    #' @param value each event in value will be checked.
    check = function(value) {
      not_implemented_error()
    },

    #' @description
    #' Define the print method for constraints,
    print = function() {
      cat(glue::glue("{class(self)}"))
    }
  )
)

#' Placeholder for variables whose support depends on other variables.
#' These variables obey no simple coordinate-wise constraints.
#'
#' @noRd
.Dependent <- R6::R6Class(
  "torch_Dependent",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    check = function(x) {
      value_error("Cannot determine validity of dependent constraint")
    }
  )
)

is_dependent <- function(object) {
  inherits(object, "torch_Dependent")
}

.IndependentConstraint <- R6::R6Class(
  "torch_IndependentConstraint",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    initialize = function(base_constraint, reinterpreted_batch_ndims) {
      self$base_constraint <- base_constraint
      self$reinterpreted_batch_ndims <- reinterpreted_batch_ndims
    },
    check = function(value) {
      result <- self$base_constraint$check(value)
      if (result$dim() < self$reinterpreted_batch_ndims) {
        expected <- self$base_constraint$event_dim + self$reinterpreted_batch_ndims
        value_error("Expected value$dim() >= {expected} but got {value$dim()}")
      }
      result <- result$reshape(c(head(result$shape, result$dim() - self$reinterpreted_batch_ndims), -1))
      result <- result$all(dim = -1)
      result
    }
  )
)

#' Constrain to the two values `{0, 1}`.
#' @noRd
.Boolean <- R6::R6Class(
  "torch_Boolean",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    check = function(value) {
      (value == 0) | (value == 1)
    }
  )
)

#' Constrain to an integer interval `[lower_bound, upper_bound]`.
#' @noRd
.IntegerInterval <- R6::R6Class(
  "torch_IntegerInterval",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    lower_bound = NULL,
    upper_bound = NULL,
    initialize = function(lower_bound, upper_bound) {
      self$lower_bound <- lower_bound
      self$upper_bound <- upper_bound
    },
    check = function(value) {
      (value %% 1 == 0) &
        (self$lower_bound <= value) &
        (value <= self$upper_bound)
    },
    print = function() {
      cat(glue::glue(
        "{class(self)}: ",
        "(lower_bound={self$lower_bound}, upper_bound={self$upper_bound})"
      ))
    }
  )
)

#' Constrain to an integer interval `(-inf, upper_bound]`.
#' @noRd
.IntegerLessThan <- R6::R6Class(
  "torch_IntegerLessThan",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    upper_bound = NULL,
    initialize = function(upper_bound) {
      self$upper_bound <- upper_bound
    },
    check = function(value) {
      (value %% 1 == 0) & (value <= self$upper_bound)
    },
    print = function() {
      cat(glue::glue(
        "{class(self)}: (upper_bound={self$upper_bound})"
      ))
    }
  )
)


#' Constrain to an integer interval `[lower_bound, inf)`.
#' @noRd
.IntegerGreaterThan <- R6::R6Class(
  "torch_IntegerGreaterThan",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    lower_bound = NULL,
    initialize = function(lower_bound) {
      self$lower_bound <- lower_bound
    },
    check = function(value) {
      (value %% 1 == 0) & (value >= self$lower_bound)
    },
    print = function() {
      cat(glue::glue(
        "{class(self)}: (lower_bound={self$lower_bound})"
      ))
    }
  )
)

#' Trivially constrain to the extended real line `[-inf, inf]`.
#' @noRd
.Real <- R6::R6Class(
  "torch_Real",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    check = function(value) {
      !torch_isnan(value)
    }
  )
)

#' Constrain to a real half line `(lower_bound, inf]`.
#' @noRd
.GreaterThan <- R6::R6Class(
  "torch_GreaterThan",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    lower_bound = NULL,
    initialize = function(lower_bound) {
      self$lower_bound <- lower_bound
    },
    check = function(value) {
      self$lower_bound < value
    },
    print = function() {
      cat(glue::glue(
        "{class(self)}: (lower_bound={self$lower_bound})"
      ))
    }
  )
)

#' Constrain to a real half line `[lower_bound, inf)`.
#' @noRd
.GreaterThanEq <- R6::R6Class(
  "torch_GreaterThanEq",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    lower_bound = NULL,
    initialize = function(lower_bound) {
      self$lower_bound <- lower_bound
    },
    check = function(value) {
      self$lower_bound <= value
    },
    print = function() {
      cat(glue::glue(
        "{class(self)}: (lower_bound={self$lower_bound})"
      ))
    }
  )
)

#' Constrain to a real half line `[lower_bound, inf)`.
#' @noRd
.LessThan <- R6::R6Class(
  "torch_LessThan",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    upper_bound = NULL,
    initialize = function(upper_bound) {
      self$upper_bound <- lower_bound
    },
    check = function(value) {
      self$upper_bound < value
    },
    print = function() {
      cat(glue::glue(
        "{class(self)}: (upper_bound={self$upper_bound})"
      ))
    }
  )
)

#' Constrain to a real interval `[lower_bound, upper_bound]`.
#' @noRd
.Interval <- R6::R6Class(
  "torch_Interval",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    lower_bound = NULL,
    upper_bound = NULL,
    initialize = function(lower_bound, upper_bound) {
      self$lower_bound <- lower_bound
      self$upper_bound <- upper_bound
    },
    check = function(value) {
      (self$lower_bound <= value) &
        (value <= self$upper_bound)
    },
    print = function() {
      cat(glue::glue(
        "(lower_bound={self$lower_bound}, upper_bound={self$upper_bound})"
      ))
    }
  )
)

#' Constrain to a real interval `[lower_bound, upper_bound)`.
#' @noRd
.HalfOpenInterval <- R6::R6Class(
  "torch_HalfOpenInterval",
  lock_objects = FALSE,
  inherit = Constraint,
  public = list(
    lower_bound = NULL,
    upper_bound = NULL,
    initialize = function(lower_bound, upper_bound) {
      self$lower_bound <- lower_bound
      self$upper_bound <- upper_bound
    },
    check = function(value) {
      (self.$ower_bound <= value) & (value < self$upper_bound)
    },
    print = function() {
      cat(glue::glue(
        "(lower_bound={self$lower_bound}, upper_bound={self$upper_bound})"
      ))
    }
  )
)

.PositiveDefinite <- R6::R6Class(
  "torch_PositiveDefinite",
  inherit = Constraint,
  public = list(
    check = function(value) {
      linalg_cholesky_ex(value)$info$eq(0)$unsqueeze(1)
    }
  )
)

.LowerCholesky <- R6::R6Class(
  "torch_LowerCholesky",
  inherit = Constraint,
  public = list(
    check = function(value) {
      value_tril <- value$tril()
      lower_triangular <- (value_tril == value)$view(c(head2(value$shape, -2), -1))$min(-1)[[1]]
      positive_diagonal <- (value$diagonal(dim1 = -2, dim2 = -1) > 0)$min(-1)[[1]]
      lower_triangular & positive_diagonal
    }
  )
)

.Simplex <- R6::R6Class(
  "torch_Simplex",
  inherit = Constraint,
  public = list(
    event_dim = 1,
    check = function(value) {
      torch_all(value >= 0, dim = -1) && ((value$sum(dim = -1) - 1)$abs() < 1e-6)
    }
  )
)

# Public interface
# TODO: check .GreaterThan and other classes,
# which are not instanced

constraint_dependent <- .Dependent$new()

constraint_boolean <- .Boolean$new()

constraint_nonnegative_integer <- .IntegerGreaterThan$new(0)

constraint_positive_integer <- .IntegerGreaterThan$new(1)

constraint_real <- .Real$new()

constraint_real_vector <- .IndependentConstraint$new(constraint_real, 1)

constraint_positive <- .GreaterThan$new(0.)

constraint_greater_than <- .GreaterThan

constraint_greater_than_eq <- .GreaterThanEq

constraint_less_than <- .LessThan

constraint_unit_interval <- .Interval$new(0., 1.)

constraint_interval <- .Interval

constraint_half_open_interval <- .HalfOpenInterval

constraint_positive_definite <- .PositiveDefinite$new()

constraint_lower_cholesky <- .LowerCholesky$new()

constraint_simplex <- .Simplex$new()
