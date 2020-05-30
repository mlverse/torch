nnf_l1_loss <- function(input, target, reduction = "mean") {
  
  if (target$requires_grad) {
    ret <- torch_abs(input - target)
    if (!is.null(reduction)) {
      
      if (reduction == "mean")
        ret <- torch_mean(ret)
      else
        ret <- torch_sum(ret)
      
    }
  } else {
    expanded <- torch_broadcast_tensors(input, target)
    ret <- torch_l1_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
  }
  
  ret
}

nnf_kl_div <- function(input, target, reduction = "mean") {
  
  if (reduction == "mean")
    warn("reduction: 'mean' divides the total loss by both the batch size and the support size.",
         "'batchmean' divides only by the batch size, and aligns with the KL div math definition.",
         "'mean' will be changed to behave the same as 'batchmean' in the next major release.")
  
  
  if (reduction == "batchmean")
    red_enum <- reduction_enum("sum")
  else
    red_enum <- reduction_enum(reduction)
  
  reduced <- torch_kl_div(input, target, red_enum)
  
  if (reduction == "batchmean")
    reduced <- reduced/ input$size(0)
  
  reduced
}

nnf_mse_loss <- function(input, target, reduction = "mean") {
  
  if (target$requires_grad) {
    ret <- (input - target) ^ 2
    if (!is.null(reduction)) {
      
      if (reduction == "mean")
        ret <- torch_mean(ret)
      else
        ret <- torch_sum(ret)
      
    }
  } else {
    
    expanded <- torch_broadcast_tensors(list(input, target))
    ret <- torch_mse_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
    
  }
  
  ret
}

nnf_binary_cross_entropy <- function(input, target, weight = NULL, size_average = NULL, 
                                     reduction = c("mean", "sum", "none")) {
  torch_binary_cross_entropy(input, target, weight, size_average=size_average,
                             reduction=reduction_enum(reduction))
}


nnf_hinge_embedding_loss <- function(input, target, margin = 1, reduction = "mean") {
  torch_hinge_embedding_loss(input, target, margin, 
                             reduction = reduction_enum(reduction))
}


nnf_multi_margin_loss <- function(input, target, p = 1, margin = 1, weight = NULL,
                                  reduction = "mean") {
  torch_multi_margin_loss(input, target, p, margin, weight, 
                          reduction = reduction_enum(reduction))
}

nnf_cosine_embedding_loss <- function(input1, input2, target, margin=0, 
                                      size_average=NULL, reduction=c("mean", "sum", "none")) {
  torch_cosine_embedding_loss(input1 = input1, input2 = input2, target = target, 
                              margin = margin, reduction = reduction_enum(reduction))
}


nnf_smooth_l1_loss <- function(input, target, reduction = "mean") {
  if (target$requires_grad) {
    ret <- nnf_smooth_l1_loss(input, target)
    if (reduction != "none") {
      if (reduction == "mean")
        ret <- torch_mean(ret)
      else
        ret <- torch_sum(ret)
    }
  } else {
    expanded <- torch_broadcast_tensors(list(input, target))
    ret <- torch_smooth_l1_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
  }
  
  ret
}

nnf_multilabel_margin_loss <- function(input, target, reduction = "mean") {
  torch_multilabel_margin_loss(input, target, reduction_enum(reduction))
}

nnf_soft_margin_loss <- function(input, target, reduction = "mean") {
  torch_soft_margin_loss(input, target, reduction_enum(reduction))
}

nnf_multilabel_soft_margin_loss <- function(input, target, weight, reduction = "mean") {
  loss <- -(target * nnf_logsigmoid(input) + (1 - target) * nnf_logsigmoid(-input))
  
  if (!is.null(weight))
    loss <- loss * weight
  
  loss <- loss$sum(dim = 1)  / input$size(1)
  
  if (reduction == "none")
    ret <- loss
  else if (reduction == "mean")
    ret <- loss$mean()
  else if (reduction == "sum")
    ret <- loss$sum()
  else
    value_error("reduction is not valid.")
  
  ret
}

nnf_triplet_margin_loss <- function(anchor, positive, negative, margin = 1, p = 2,
                                    eps = 1e-6, swap = FALSE, reduction = "mean") {
  torch_triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap,
                            reduction_enum(reduction))
}

nnf_ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank=0,
                         reduction=c('mean', "sum", "none"), zero_infinity=FALSE) {
  torch_ctc_loss(log_probs = log_probs, targets = targets, input_lengths = input_lengths,
                 target_lengths = target_lengths, blank = blank, reduction = reduction_enum(reduction),
                 zero_infinity = zero_infinity)
}


nnf_poisson_nll_loss <- function(input, target, log_input = TRUE, full = FALSE, 
                                 eps = 1e-8, reduction = "mean") {
  torch_poisson_nll_loss(input, target, log_input, full, eps, 
                         reduction_enum(reduction))
}

nnf_margin_ranking_loss <- function(input1, input2, target, margin = 0,
                                    reduction = "mean") {
  torch_margin_ranking_loss(input1, input2, target, margin, 
                            reduction_enum(reduction))
}

nnf_nll_loss <- function(input, target, weight = NULL, ignore_index, 
                         reduction = "mean") {
  
  dim <- input$dim()
  
  if (dim < 2)
    value_error("Expected 2 or more dimensions, got '{dim}'.")
  
  if (dim == 2)
    ret <- torch_nll_loss(input, target, weight, reduction_enum(reduction), 
                          ignore_index)
  else if (dim == 4)
    ret <- torch_nll_loss2d(input, target, weight, reduction_enum(reduction),
                            ignore_index)
  else {
    n <- input$size(0)
    c <- input$size(1)
    out_size <- c(n, input$size()[-c(1:2)])
    
    input <- input$contiguous()
    target <- target$contiguous()
    
    if (input$numel() > 0)
      input <- input$view(n, c, 1, -1)
    else
      input <- input$view(n, c, 0, 0)
    
    if (target$numel() > 0)
      target <- target$view(n, 1, -1)
    else
      target <- target$view(n, 0, 0)
    
    if (reduction != "none") {
      ret <- torch_nll_loss2d(input, target, weight, reduction_enum(reduction),
                              ignore_index)
    } else {
      out <- torch_nll_loss2d(input, target, weight, reduction_enum(reduction),
                              ignore_index)
      ret <- out$view(out_size)
    }
  }
  
  ret
}

nnf_cross_entropy <- function(input, target, weight=NULL, ignore_index=-100, 
                              reduction=c("mean", "sum", "none")) {
  torch_nll_loss(self = torch_log_softmax(input, 1), target = target, weight = weight, 
                 reduction = reduction_enum(reduction), ignore_index = ignore_index)
}

nnf_binary_cross_entropy_with_logits <- function(input, target, weight = NULL, 
                                                 size_average = NULL, 
                                                 reduction = c("mean", "sum", "none"), 
                                                 pos_weight = NULL) {
  torch_binary_cross_entropy_with_logits(input, target, weigth, pos_weight, 
                                         reduction_enum(reduction))
}
