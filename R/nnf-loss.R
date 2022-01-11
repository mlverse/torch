#' L1_loss
#'
#' Function that takes the mean element-wise absolute value difference.
#'
#' @param input tensor (N,*) where ** means, any number of additional dimensions
#' @param target tensor (N,*) , same shape as the input
#' @param reduction (string, optional) – Specifies the reduction to apply to the
#'   output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean':
#'   the sum of the output will be divided by the number of elements in the output,
#'   'sum': the output will be summed. Default: 'mean'
#'
#' @export
nnf_l1_loss <- function(input, target, reduction = "mean") {
  if (target$requires_grad) {
    ret <- torch_abs(input - target)
    if (!is.null(reduction)) {
      if (reduction == "mean") {
        ret <- torch_mean(ret)
      } else {
        ret <- torch_sum(ret)
      }
    }
  } else {
    expanded <- torch_broadcast_tensors(list(input, target))
    ret <- torch_l1_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
  }

  ret
}

#' Kl_div
#'
#' The Kullback-Leibler divergence Loss.
#'
#' @inheritParams nnf_l1_loss
#'
#' @export
nnf_kl_div <- function(input, target, reduction = "mean") {
  if (reduction == "mean") {
    warn(
      "reduction: 'mean' divides the total loss by both the batch size and the support size.",
      "'batchmean' divides only by the batch size, and aligns with the KL div math definition.",
      "'mean' will be changed to behave the same as 'batchmean' in the next major release."
    )
  }


  if (reduction == "batchmean") {
    red_enum <- reduction_enum("sum")
  } else {
    red_enum <- reduction_enum(reduction)
  }

  reduced <- torch_kl_div(input, target, red_enum)

  if (reduction == "batchmean") {
    reduced <- reduced / input$size(1)
  }

  reduced
}

#' Mse_loss
#'
#' Measures the element-wise mean squared error.
#'
#' @inheritParams nnf_l1_loss
#'
#' @export
nnf_mse_loss <- function(input, target, reduction = "mean") {
  if (!all(target$shape == input$shape)) {
    target_shape <- paste0("(", paste(target$shape, collapse = ","), ")")
    input_shape <- paste0("(", paste(input$shape, collapse = ","), ")")

    warn(
      "Using a target size {target_shape} that is different to the input size {input_shape}. ",
      "This will likely lead to incorrect results due to broadcasting. ",
      "Please ensure they have the same size."
    )
  }

  if (target$requires_grad) {
    ret <- (input - target)^2
    if (!is.null(reduction)) {
      if (reduction == "mean") {
        ret <- torch_mean(ret)
      } else {
        ret <- torch_sum(ret)
      }
    }
  } else {
    expanded <- torch_broadcast_tensors(list(input, target))
    ret <- torch_mse_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
  }

  ret
}

#' Binary_cross_entropy
#'
#' Function that measures the Binary Cross Entropy
#' between the target and the output.
#'
#' @param weight (tensor) weight for each value.
#' @inheritParams nnf_l1_loss
#'
#' @export
nnf_binary_cross_entropy <- function(input, target, weight = NULL,
                                     reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)
  torch_binary_cross_entropy(input, target, weight,
    reduction = reduction_enum(reduction)
  )
}

#' Hinge_embedding_loss
#'
#' Measures the loss given an input tensor xx and a labels tensor yy (containing 1 or -1).
#' This is usually used for measuring whether two inputs are similar or dissimilar, e.g.
#' using the L1 pairwise distance as xx , and is typically used for learning nonlinear
#' embeddings or semi-supervised learning.
#'
#' @inheritParams nnf_l1_loss
#' @param margin Has a default value of 1.
#'
#' @export
nnf_hinge_embedding_loss <- function(input, target, margin = 1, reduction = "mean") {
  torch_hinge_embedding_loss(input, target, margin,
    reduction = reduction_enum(reduction)
  )
}

#' Multi_margin_loss
#'
#' Creates a criterion that optimizes a multi-class classification hinge loss
#' (margin-based loss) between input x (a 2D mini-batch Tensor) and output y
#' (which is a 1D tensor of target class indices, `0 <= y <= x$size(2) - 1` ).
#'
#' @inheritParams nnf_l1_loss
#' @param p Has a default value of 1. 1 and 2 are the only supported values.
#' @param margin Has a default value of 1.
#' @param weight a manual rescaling weight given to each class. If given, it has to
#'   be a Tensor of size C. Otherwise, it is treated as if having all ones.
#'
#' @export
nnf_multi_margin_loss <- function(input, target, p = 1, margin = 1, weight = NULL,
                                  reduction = "mean") {
  torch_multi_margin_loss(input, target, p, margin, weight,
    reduction = reduction_enum(reduction)
  )
}

#' Cosine_embedding_loss
#'
#' Creates a criterion that measures the loss given input tensors x_1, x_2 and a
#' Tensor label y with values 1 or -1. This is used for measuring whether two inputs
#' are similar or dissimilar, using the cosine distance, and is typically used
#' for learning nonlinear embeddings or semi-supervised learning.
#'
#' @inheritParams nnf_l1_loss
#' @param input1 the input x_1 tensor
#' @param input2 the input x_2 tensor
#' @param target the target tensor
#' @param margin Should be a number from -1 to 1 , 0 to 0.5 is suggested. If margin
#'   is missing, the default value is 0.
#'
#' @export
nnf_cosine_embedding_loss <- function(input1, input2, target, margin = 0,
                                      reduction = c("mean", "sum", "none")) {
  torch_cosine_embedding_loss(
    input1 = input1, input2 = input2, target = target,
    margin = margin, reduction = reduction_enum(reduction)
  )
}

#' Smooth_l1_loss
#'
#' Function that uses a squared term if the absolute
#' element-wise error falls below 1 and an L1 term otherwise.
#'
#' @inheritParams nnf_l1_loss
#'
#' @export
nnf_smooth_l1_loss <- function(input, target, reduction = "mean") {
  expanded <- torch_broadcast_tensors(list(input, target))
  torch_smooth_l1_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
}

#' Multilabel_margin_loss
#'
#' Creates a criterion that optimizes a multi-class multi-classification hinge loss
#' (margin-based loss) between input x (a 2D mini-batch Tensor) and output y (which
#' is a 2D Tensor of target class indices).
#'
#' @inheritParams nnf_l1_loss
#'
#' @export
nnf_multilabel_margin_loss <- function(input, target, reduction = "mean") {
  torch_multilabel_margin_loss(input, target, reduction_enum(reduction))
}

#' Soft_margin_loss
#'
#' Creates a criterion that optimizes a two-class classification logistic loss
#' between input tensor x and target tensor y (containing 1 or -1).
#'
#' @inheritParams nnf_l1_loss
#'
#' @export
nnf_soft_margin_loss <- function(input, target, reduction = "mean") {
  torch_soft_margin_loss(input, target, reduction_enum(reduction))
}

#' Multilabel_soft_margin_loss
#'
#' Creates a criterion that optimizes a multi-label one-versus-all loss based on
#' max-entropy, between input x and target y of size (N, C).
#'
#' @inheritParams nnf_l1_loss
#' @param weight weight tensor to apply on the loss.
#'
#' @export
nnf_multilabel_soft_margin_loss <- function(input, target, weight = NULL, reduction = "mean") {
  loss <- -(target * nnf_logsigmoid(input) + (1 - target) * nnf_logsigmoid(-input))

  if (!is.null(weight)) {
    loss <- loss * weight
  }

  loss <- loss$sum(dim = 1) / input$size(2)

  if (reduction == "none") {
    ret <- loss
  } else if (reduction == "mean") {
    ret <- loss$mean()
  } else if (reduction == "sum") {
    ret <- loss$sum()
  } else {
    value_error("reduction is not valid.")
  }

  ret
}

#' Triplet_margin_loss
#'
#' Creates a criterion that measures the triplet loss given an input tensors x1 ,
#' x2 , x3 and a margin with a value greater than 0 . This is used for measuring
#' a relative similarity between samples. A triplet is composed by a, p and n (i.e.,
#' anchor, positive examples and negative examples respectively). The shapes of all
#' input tensors should be (N, D).
#'
#' @inheritParams nnf_l1_loss
#' @param anchor the anchor input tensor
#' @param positive the positive input tensor
#' @param negative the negative input tensor
#' @param margin Default: 1.
#' @param p The norm degree for pairwise distance. Default: 2.
#' @param eps (float, optional) Small value to avoid division by zero.
#' @param swap The distance swap is described in detail in the paper Learning shallow
#'   convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.
#'   Default: `FALSE`.
#'
#' @export
nnf_triplet_margin_loss <- function(anchor, positive, negative, margin = 1, p = 2,
                                    eps = 1e-6, swap = FALSE, reduction = "mean") {
  torch_triplet_margin_loss(
    anchor, positive, negative, margin, p, eps, swap,
    reduction_enum(reduction)
  )
}

#' Triplet margin with distance loss
#'
#' See [nn_triplet_margin_with_distance_loss()]
#'
#' @inheritParams nnf_triplet_margin_loss
#' @inheritParams nn_triplet_margin_with_distance_loss
#'
#' @export
nnf_triplet_margin_with_distance_loss <- function(anchor, positive, negative,
                                                  distance_function = NULL,
                                                  margin = 1.0, swap = FALSE,
                                                  reduction = "mean") {
  if (is.null(distance_function)) {
    distance_function <- nnf_pairwise_distance
  }

  positive_dist <- distance_function(anchor, positive)
  negative_dist <- distance_function(anchor, negative)

  if (swap) {
    swap_dist <- distance_function(positive, negative)
    negative_dist <- torch_min(negative_dist, swap_dist)
  }

  output <- torch_clamp(positive_dist - negative_dist + margin, min = 0.0)

  reduction_enum <- reduction_enum(reduction)

  if (reduction_enum == 1) {
    return(output$mean())
  } else if (reduction_enum == 2) {
    return(output$sum())
  } else {
    return(output)
  }
}

#' Ctc_loss
#'
#' The Connectionist Temporal Classification loss.
#'
#' @inheritParams nnf_l1_loss
#' @param log_probs \eqn{(T, N, C)} where C = number of characters in alphabet including blank,
#'   T = input length, and N = batch size. The logarithmized probabilities of
#'   the outputs (e.g. obtained with [nnf_log_softmax]).
#' @param targets \eqn{(N, S)} or `(sum(target_lengths))`. Targets cannot be blank.
#'   In the second form, the targets are assumed to be concatenated.
#' @param input_lengths \eqn{(N)}. Lengths of the inputs (must each be \eqn{\leq T})
#' @param target_lengths \eqn{(N)}. Lengths of the targets
#' @param blank (int, optional) Blank label. Default \eqn{0}.
#' @param zero_infinity (bool, optional) Whether to zero infinite losses and the
#'   associated gradients. Default: `FALSE` Infinite losses mainly occur when the
#'   inputs are too short to be aligned to the targets.
#'
#' @export
nnf_ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank = 0,
                         reduction = c("mean", "sum", "none"), zero_infinity = FALSE) {
  torch_ctc_loss(
    log_probs = log_probs, targets = targets, input_lengths = input_lengths,
    target_lengths = target_lengths, blank = blank, reduction = reduction_enum(reduction),
    zero_infinity = zero_infinity
  )
}

#' Poisson_nll_loss
#'
#' Poisson negative log likelihood loss.
#'
#' @inheritParams nnf_l1_loss
#' @param log_input if `TRUE` the loss is computed as \eqn{\exp(\mbox{input}) - \mbox{target} * \mbox{input}},
#'   if `FALSE` then loss is \eqn{\mbox{input} - \mbox{target} * \log(\mbox{input}+\mbox{eps})}.
#'   Default: `TRUE`.
#' @param full whether to compute full loss, i. e. to add the Stirling approximation
#'  term. Default: `FALSE`.
#' @param eps (float, optional) Small value to avoid evaluation of \eqn{\log(0)} when
#'   `log_input`=`FALSE`. Default: 1e-8
#'
#' @export
nnf_poisson_nll_loss <- function(input, target, log_input = TRUE, full = FALSE,
                                 eps = 1e-8, reduction = "mean") {
  torch_poisson_nll_loss(
    input, target, log_input, full, eps,
    reduction_enum(reduction)
  )
}

#' Margin_ranking_loss
#'
#' Creates a criterion that measures the loss given inputs x1 , x2 , two 1D
#' mini-batch Tensors, and a label 1D mini-batch tensor y (containing 1 or -1).
#'
#' @inheritParams nnf_l1_loss
#' @param input1 the first tensor
#' @param input2 the second input tensor
#' @param target the target tensor
#' @param margin Has a default value of 00 .
#'
#' @export
nnf_margin_ranking_loss <- function(input1, input2, target, margin = 0,
                                    reduction = "mean") {
  torch_margin_ranking_loss(
    input1, input2, target, margin,
    reduction_enum(reduction)
  )
}

#' Nll_loss
#'
#' The negative log likelihood loss.
#'
#' @inheritParams nnf_l1_loss
#' @param input \eqn{(N, C)} where `C = number of classes` or \eqn{(N, C, H, W)} in
#'   case of 2D Loss, or \eqn{(N, C, d_1, d_2, ..., d_K)} where \eqn{K \geq 1} in
#'   the case of K-dimensional loss.
#' @param target \eqn{(N)} where each value is \eqn{0 \leq \mbox{targets}[i] \leq C-1},
#'   or \eqn{(N, d_1, d_2, ..., d_K)} where \eqn{K \geq 1} for K-dimensional loss.
#' @param weight (Tensor, optional) a manual rescaling weight given to each class.
#'   If given, has to be a Tensor of size `C`
#' @param ignore_index (int, optional) Specifies a target value that is ignored and
#'   does not contribute to the input gradient.
#'
#' @export
nnf_nll_loss <- function(input, target, weight = NULL, ignore_index = -100,
                         reduction = "mean") {
  dim <- input$dim()

  if (dim < 2) {
    value_error("Expected 2 or more dimensions, got '{dim}'.")
  }

  if (dim == 2) {
    ret <- torch_nll_loss(
      input, target, weight, reduction_enum(reduction),
      ignore_index
    )
  } else if (dim == 4) {
    ret <- torch_nll_loss2d(
      input, target, weight, reduction_enum(reduction),
      ignore_index
    )
  } else {
    n <- input$size(1)
    c <- input$size(2)
    out_size <- c(n, input$size()[-c(1:2)])

    input <- input$contiguous()
    target <- target$contiguous()

    if (input$numel() > 0) {
      input <- input$view(c(n, c, 1, -1))
    } else {
      input <- input$view(c(n, c, 0, 0))
    }

    if (target$numel() > 0) {
      target <- target$view(c(n, 1, -1))
    } else {
      target <- target$view(c(n, 0, 0))
    }

    if (reduction != "none") {
      ret <- torch_nll_loss2d(
        input, target, weight, reduction_enum(reduction),
        ignore_index
      )
    } else {
      out <- torch_nll_loss2d(
        input, target, weight, reduction_enum(reduction),
        ignore_index
      )
      ret <- out$view(out_size)
    }
  }

  ret
}

#' Cross_entropy
#'
#' This criterion combines `log_softmax` and `nll_loss` in a single
#' function.
#'
#' @inheritParams nnf_l1_loss
#' @param input (Tensor) \eqn{(N, C)} where `C = number of classes` or \eqn{(N, C, H, W)}
#'   in case of 2D Loss, or \eqn{(N, C, d_1, d_2, ..., d_K)} where \eqn{K \geq 1}
#'   in the case of K-dimensional loss.
#' @param target (Tensor) \eqn{(N)} where each value is \eqn{0 \leq \mbox{targets}[i] \leq C-1},
#'   or \eqn{(N, d_1, d_2, ..., d_K)} where \eqn{K \geq 1} for K-dimensional loss.
#' @param weight (Tensor, optional) a manual rescaling weight given to each class. If
#'   given, has to be a Tensor of size `C`
#' @param ignore_index (int, optional) Specifies a target value that is ignored
#'   and does not contribute to the input gradient.
#'
#' @export
nnf_cross_entropy <- function(input, target, weight = NULL, ignore_index = -100,
                              reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)
  torch_cross_entropy_loss(
    self = input, target = target, weight = weight,
    reduction = reduction_enum(reduction),
    ignore_index = ignore_index
  )
}

#' Binary_cross_entropy_with_logits
#'
#' Function that measures Binary Cross Entropy between target and output
#' logits.
#'
#' @inheritParams nnf_l1_loss
#' @param input Tensor of arbitrary shape
#' @param target Tensor of the same shape as input
#' @param weight (Tensor, optional) a manual rescaling weight if provided it's
#'   repeated to match input tensor shape.
#' @param pos_weight (Tensor, optional) a weight of positive examples.
#'   Must be a vector with length equal to the number of classes.
#'
#' @export
nnf_binary_cross_entropy_with_logits <- function(input, target, weight = NULL,
                                                 reduction = c("mean", "sum", "none"),
                                                 pos_weight = NULL) {
  reduction <- match.arg(reduction)
  torch_binary_cross_entropy_with_logits(
    input, target, weight, pos_weight,
    reduction_enum(reduction)
  )
}
