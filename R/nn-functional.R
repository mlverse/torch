nnf_adaptive_avg_pool1d <- function(input, output_size) {
  torch_adaptive_avg_pool1d(input, output_size)
}

nnf_adaptive_avg_pool2d <- function(input, output_size) {
  torch_adaptive_avg_pool2d(input, output_size)
}

nnf_adaptive_avg_pool3d <- function(input, output_size) {
  torch_adaptive_avg_pool3d(input, output_size)
}

nnf_adaptive_max_pool1d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool1d(input, output_size)
  if (!return_indices)
    o <- o[[1]]
  
  o
}

nnf_adaptive_max_pool1d_with_indices <- function(input, output_size) {
  torch_adaptive_max_pool1d(input, output_size)
}

nnf_adaptive_max_pool2d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool2d(input, output_size)
  if (!return_indices)
    o <- o[[1]]
  
  o
}

nnf_adaptive_max_pool2d_with_indices <- function(input, output_size) {
  torch_adaptive_max_pool2d(input, output_size)
}

nnf_adaptive_max_pool3d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool3d(input, output_size)
  if (!return_indices)
    o <- o[[1]]
  
  o
}

nnf_adaptive_max_pool3d_with_indices <- function(input, output_size) {
  torch_adaptive_max_pool3d(input, output_size)
}

nnf_affine_grid <- function(theta, size, align_corners = FALSE) {
  torch_affine_grid_generator(theta, size, align_corners)
}

nnf_alpha_dropout <- function(input, p = 0.5, training = FALSE, inplace = FALSE) {
  if (inplace)
    torch_alpha_dropout_(input, p, training)
  else
    torch_alpha_dropout(input, p, training)
}

nnf_avg_pool1d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE, 
                           count_include_pad = TRUE) {
  torch_avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
}

nnf_avg_pool2d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE, 
                           count_include_pad = TRUE, divisor_override = NULL) {
  torch_avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad,
                   divisor_override)
}

nnf_avg_pool3d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE, 
                           count_include_pad = TRUE, divisor_override = NULL) {
  torch_avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad,
                   divisor_override)
}

nnf_batch_norm <- function(input, running_mean, running_var, weight = NULL, bias = NULL,
                           training = FALSE, momentum = 0.1, eps = 1e-5) {
  torch_batch_norm(input = input, weight = weight, bias = bias, running_mean = running_mean,
                   running_var = running_var, training = training, momentum = momentum,
                   eps = eps, cudnn_enabled = backends_cudnn_enabled())
}

nnf_bilinear <- function(input1, input2, weight, bias = NULL) {
  torch_bilinear(input1, input2, weight, bias)
}

nnf_binary_cross_entropy <- function(input, target, weight = NULL, size_average = NULL, 
                                     reduction = c("mean", "sum", "none")) {
  torch_binary_cross_entropy(input, target, weight, size_average=size_average,
                             reduction=reduction_enum(reduction))
}

nnf_binary_cross_entropy_with_logits <- function(input, target, weight = NULL, 
                                                 size_average = NULL, 
                                                 reduction = c("mean", "sum", "none"), 
                                                 pos_weight = NULL) {
  torch_binary_cross_entropy_with_logits(input, target, weigth, pos_weight, 
                                         reduction_enum(reduction))
}

nnf_celu <- function(input, alpha = 1, inplace = FALSE) {
  if (inplace)
    torch_celu_(input, alpha)
  else
    torch_celu(input, alpha)
}

nnf_celu_ <- function(input, alpha = 1) {
  torch_celu_(input, alpha)
}

nnf_conv1d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv1d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

nnf_conv2d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv2d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

nnf_conv3d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv3d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

nnf_conv_tbc <- function(input, weight, bias, pad = 0) {
  torch_conv_tbc(self = input, weight = weight, bias = bias, pad = pad)
}

nnf_conv_transpose1d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose1d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_conv_transpose2d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose2d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_conv_transpose3d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose3d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_cosine_embedding_loss <- function(input1, input2, target, margin=0, 
                                      size_average=NULL, reduction=c("mean", "sum", "none")) {
  torch_cosine_embedding_loss(input1 = input1, input2 = input2, target = target, 
                              margin = margin, reduction = reduction_enum(reduction))
}

nnf_cosine_similarity <- function(x1, x2, dim=1, eps=1e-8) {
  torch_cosine_similarity(x1 = x1, x2 = x2, dim = dim, eps = eps)
}

nnf_cross_entropy <- function(input, target, weight=NULL, ignore_index=-100, 
                              reduction=c("mean", "sum", "none")) {
  torch_nll_loss(self = torch_log_softmax(input, 1), target = target, weight = weight, 
                 reduction = reduction_enum(reduction), ignore_index = ignore_index)
}

nnf_ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank=0,
                         reduction=c('mean', "sum", "none"), zero_infinity=FALSE) {
  torch_ctc_loss(log_probs = log_probs, targets = targets, input_lengths = input_lengths,
                 target_lengths = target_lengths, blank = blank, reduction = reduction_enum(reduction),
                 zero_infinity = zero_infinity)
}

nnf_dropout <- function(input, p=0.5, training=TRUE, inplace=FALSE) {
  if (inplace)
    torch_dropout_(input, p, training)
  else
    torch_dropout(input, p, training)
}

nnf_dropout2d <- function(input, p=0.5, training=TRUE, inplace=FALSE) {
  if (inplace)
    torch_feature_dropout_(input, p, training)
  else
    torch_feature_dropout(input, p, training)
}

nnf_dropout3d <- function(input, p=0.5, training=TRUE, inplace=FALSE) {
  if (inplace)
    torch_feature_dropout_(input, p, training)
  else
    torch_feature_dropout(input, p, training)
}

nnf_elu <- function(input, alpha=1, inplace=FALSE) {
  if(inplace)
    torch_elu_(input, alpha = alpha)
  else
    torch_elu(input, alpha = alpha)
}

nnf_elu_ <- function(input, alpha=1) {
  torch_elu_(input, alpha = alpha)
}

nnf_embedding <- function(input, weight, padding_idx=NULL, max_norm=NULL, norm_type=2,
                          scale_grad_by_freq=FALSE, sparse=FALSE) {
  if (is.null(padding_idx))
    padding_idx <- -1
  
  if (!is.null(max_norm)) {
    input <- input$contiguous()
    with_no_grad({
      torch_embedding_renorm_(weight, input, max_norm, norm_type)
    })
  }
  
  torch_embedding(weight = weight, input = input, padding_idx = padding_idx,
                  scale_grad_by_freq = scale_grad_by_freq, sparse = sparse)  
}

nnf_embedding_bag <- function(input, weight, offsets = NULL, max_norm = NULL, 
                              norm_type = 2, scale_grad_by_freq = FALSE, 
                              mode = "mean", sparse= FALSE, per_sample_weights = NULL,
                              include_last_offset = FALSE) {

  if (input$dim() == 2) {
    input <- input$reshape(-1)
    if (!is.null(per_sample_weights)) {
      per_sample_weights <- per_sample_weights$reshape(-1)
    }
  } 
  
  if (mode == 'sum') {
    mode_enum <- 0
  } else if (mode == "mean") {
    mode_enum <- 1
  } else if (mode == "max") {
    mode_enum <- 2
  }
    
  if (!is.null(max_norm)) {
    input <- input$contiguous()
    with_no_grad({
      torch_embedding_renorm_(weight, input, max_norm, norm_type)
    })
  }
  
  ret <- torch_embedding_bag(weight = weight, indices = input, offsets = offsets, 
                      scale_grad_by_freq = scale_grad_by_freq, mode = mode_enum,
                      sparse = sparse, per_sample_weights = per_sample_weights, 
                      include_last_offset = include_last_offset)
                      
  ret[[1]]
}

pair <- function(x) {
  if (length(x) == 1)
    rep(x, 2)
  else
    x
}

nnf_fold <- function(input, output_size, kernel_size, dilation=1, padding=0, stride=1) {
  torch_col2im(self = input, output_size = pair(output_size), 
               kernel_size = pair(kernel_size), dilation = pair(dilation), 
               padding = pair(padding), stride = pair(stride))
}

nnf_fractional_max_pool2d <- function(input, kernel_size, output_size=NULL,
                                      output_ratio=NULL, return_indices=FALSE,
                                      random_samples = NULL) {
  
  if (is.null(output_size)) {
    output_ratio_ <- pair(output_ratio)
    output_size <- list(
      as.integer(input$size(2) * output_ratio_[[1]]),
      as.integer(input$size(3) * output_ratio_[[2]])
    )
  }
  
  if (is.null(random_samples)) {
    random_samples <- torch_rand(input$size(0), input$size(1), 2, 
                                 dtype = input$dtype, device = input$device)
  }
  
  res <- torch_fractional_max_pool2d(self = input, kernel_size = kernel_size, 
                              output_size = output_size, random_samples = random_samples)
  
  if (return_indices)
    res
  else
    res[[1]]
}


nnf_fractional_max_pool3d <- function(kernel_size, output_size = NULL, output_ratio = NULL, 
                                      return_indices = FALSE, random_samples = NULL) {
  
  
  if (is.null(output_size)) {
    
    if (length(output_size) == 1)
      
      output_ratio_ <- rep(output_size, 3)
      output_size <- c(
        input$size(2) * output_ratio_[1],
        input$size(3) * output_ratio_[2],
        input$size(4) * output_ratio_[3],
      )
  }
  
  if (is.null(random_samples)) {
    random_samples <- torch_rand(input$size(0), input$size(1), 3, dtype = input$dtype, 
                                 device = input$device)
  }
  
  res <- torch_fractional_max_pool3d(
    self = input, 
    kernel_size = kernel_size,
    output_size = output_size,
    random_samples = random_samples
  )
  
  
  if (return_indices)
    res
  else
    res[[1]]
}

nnf_gelu <- function(input) {
  torch_gelu(self = input)
}

nnf_glu <- function(input, dim = -1) {
  torch_glu(self = input, dim = dim)
}

nnf_grad <- function() {
# """Gradient interface"""
# 
# import torch
# from .modules.utils import _single, _pair, _triple
# 
# 
# def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size):
#     input_size = list(input_size)
#     k = grad_output.dim() - 2
# 
#     if len(input_size) == k + 2:
#         input_size = input_size[-k:]
#     if len(input_size) != k:
#         raise ValueError("input_size must have {} elements (got {})"
#                          .format(k + 2, len(input_size)))
# 
#     def dim_size(d):
#         return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
#                 kernel_size[d])
# 
#     min_sizes = [dim_size(d) for d in range(k)]
#     max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
#     for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
#         if size < min_size or size > max_size:
#             raise ValueError(
#                 ("requested an input grad size of {}, but valid sizes range "
#                  "from {} to {} (for a grad_output of {})").format(
#                      input_size, min_sizes, max_sizes,
#                      grad_output.size()[2:]))
# 
#     return tuple(input_size[d] - min_sizes[d] for d in range(k))
# 
# 
# def conv1d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
#     r"""
#     Computes the gradient of conv1d with respect to the input of the convolution.
#     This is same as the 1D transposed convolution operator under the hood but requires
#     the shape of the gradient w.r.t. input to be specified explicitly.
# 
#     Args:
#         input_size : Shape of the input gradient tensor
#         weight: weight tensor (out_channels x in_channels/groups x kW)
#         grad_output : output gradient tensor (minibatch x out_channels x oW)
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
# 
#     Examples::
# 
#         >>> input = torch.randn(1,1,3, requires_grad=True)
#         >>> weight = torch.randn(1,1,1, requires_grad=True)
#         >>> output = F.conv1d(input, weight)
#         >>> grad_output = torch.randn(output.shape)
#         >>> grad_input = torch.autograd.grad(output, input, grad_output)
#         >>> F.grad.conv1d_input(input.shape, weight, grad_output)
# 
#     """
#     stride = _single(stride)
#     padding = _single(padding)
#     dilation = _single(dilation)
#     kernel_size = [weight.shape[2]]
# 
#     if input_size is None:
#         raise ValueError("grad.conv1d_input requires specifying an input_size")
# 
#     grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
#                                              padding, kernel_size)
# 
#     return torch.conv_transpose1d(
#         grad_output, weight, None, stride, padding, grad_input_padding, groups,
#         dilation)
# 
# 
# def conv1d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
#     r"""
#     Computes the gradient of conv1d with respect to the weight of the convolution.
# 
#     Args:
#         input: input tensor of shape (minibatch x in_channels x iW)
#         weight_size : Shape of the weight gradient tensor
#         grad_output : output gradient tensor (minibatch x out_channels x oW)
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
# 
#     Examples::
# 
#         >>> input = torch.randn(1,1,3, requires_grad=True)
#         >>> weight = torch.randn(1,1,1, requires_grad=True)
#         >>> output = F.conv1d(input, weight)
#         >>> grad_output = torch.randn(output.shape)
#         >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
#         >>> F.grad.conv1d_weight(input, weight.shape, grad_output)
# 
#     """
#     stride = _single(stride)
#     padding = _single(padding)
#     dilation = _single(dilation)
#     in_channels = input.shape[1]
#     out_channels = grad_output.shape[1]
#     min_batch = input.shape[0]
# 
#     grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1)
#     grad_output = grad_output.contiguous().view(
#         grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2])
# 
#     input = input.contiguous().view(1, input.shape[0] * input.shape[1],
#                                     input.shape[2])
# 
#     grad_weight = torch.conv1d(input, grad_output, None, dilation, padding,
#                                stride, in_channels * min_batch)
# 
#     grad_weight = grad_weight.contiguous().view(
#         min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2])
# 
#     return grad_weight.sum(dim=0).view(
#         in_channels // groups, out_channels, grad_weight.shape[2]).transpose(
#             0, 1).narrow(2, 0, weight_size[2])
# 
# 
# def conv2d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
#     r"""
#     Computes the gradient of conv2d with respect to the input of the convolution.
#     This is same as the 2D transposed convolution operator under the hood but requires
#     the shape of the gradient w.r.t. input to be specified explicitly.
# 
#     Args:
#         input_size : Shape of the input gradient tensor
#         weight: weight tensor (out_channels x in_channels/groups x kH x kW)
#         grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
# 
#     Examples::
# 
#         >>> input = torch.randn(1,1,3,3, requires_grad=True)
#         >>> weight = torch.randn(1,1,1,2, requires_grad=True)
#         >>> output = F.conv2d(input, weight)
#         >>> grad_output = torch.randn(output.shape)
#         >>> grad_input = torch.autograd.grad(output, input, grad_output)
#         >>> F.grad.conv2d_input(input.shape, weight, grad_output)
# 
#     """
#     stride = _pair(stride)
#     padding = _pair(padding)
#     dilation = _pair(dilation)
#     kernel_size = (weight.shape[2], weight.shape[3])
# 
#     if input_size is None:
#         raise ValueError("grad.conv2d_input requires specifying an input_size")
# 
#     grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
#                                              padding, kernel_size)
# 
#     return torch.conv_transpose2d(
#         grad_output, weight, None, stride, padding, grad_input_padding, groups,
#         dilation)
# 
# 
# def conv2d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
#     r"""
#     Computes the gradient of conv2d with respect to the weight of the convolution.
# 
#     Args:
#         input: input tensor of shape (minibatch x in_channels x iH x iW)
#         weight_size : Shape of the weight gradient tensor
#         grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
# 
#     Examples::
# 
#         >>> input = torch.randn(1,1,3,3, requires_grad=True)
#         >>> weight = torch.randn(1,1,1,2, requires_grad=True)
#         >>> output = F.conv2d(input, weight)
#         >>> grad_output = torch.randn(output.shape)
#         >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
#         >>> F.grad.conv2d_weight(input, weight.shape, grad_output)
# 
#     """
#     stride = _pair(stride)
#     padding = _pair(padding)
#     dilation = _pair(dilation)
#     in_channels = input.shape[1]
#     out_channels = grad_output.shape[1]
#     min_batch = input.shape[0]
# 
#     grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1,
#                                                   1)
#     grad_output = grad_output.contiguous().view(
#         grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
#         grad_output.shape[3])
# 
#     input = input.contiguous().view(1, input.shape[0] * input.shape[1],
#                                     input.shape[2], input.shape[3])
# 
#     grad_weight = torch.conv2d(input, grad_output, None, dilation, padding,
#                                stride, in_channels * min_batch)
# 
#     grad_weight = grad_weight.contiguous().view(
#         min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
#         grad_weight.shape[3])
# 
#     return grad_weight.sum(dim=0).view(
#         in_channels // groups, out_channels,
#         grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
#             2, 0, weight_size[2]).narrow(3, 0, weight_size[3])
# 
# 
# def conv3d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
#     r"""
#     Computes the gradient of conv3d with respect to the input of the convolution.
#     This is same as the 3D transposed convolution operator under the hood but requires
#     the shape of the gradient w.r.t. input to be specified explicitly.
# 
#     Args:
#         input_size : Shape of the input gradient tensor
#         weight: weights tensor (out_channels x in_channels/groups x kT x kH x kW)
#         grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
# 
#     Examples::
# 
#         >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
#         >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
#         >>> output = F.conv3d(input, weight)
#         >>> grad_output = torch.randn(output.shape)
#         >>> grad_input = torch.autograd.grad(output, input, grad_output)
#         >>> F.grad.conv3d_input(input.shape, weight, grad_output)
# 
#     """
#     stride = _triple(stride)
#     padding = _triple(padding)
#     dilation = _triple(dilation)
#     kernel_size = (weight.shape[2], weight.shape[3], weight.shape[4])
# 
#     if input_size is None:
#         raise ValueError("grad.conv3d_input requires specifying an input_size")
# 
#     grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
#                                              padding, kernel_size)
# 
#     return torch.conv_transpose3d(
#         grad_output, weight, None, stride, padding, grad_input_padding, groups,
#         dilation)
# 
# 
# def conv3d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
#     r"""
#     Computes the gradient of conv3d with respect to the weight of the convolution.
# 
#     Args:
#         input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
#         weight_size : Shape of the weight gradient tensor
#         grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
# 
#     Examples::
# 
#         >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
#         >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
#         >>> output = F.conv3d(input, weight)
#         >>> grad_output = torch.randn(output.shape)
#         >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
#         >>> F.grad.conv3d_weight(input, weight.shape, grad_output)
# 
#     """
#     stride = _triple(stride)
#     padding = _triple(padding)
#     dilation = _triple(dilation)
#     in_channels = input.shape[1]
#     out_channels = grad_output.shape[1]
#     min_batch = input.shape[0]
# 
#     grad_output = grad_output.repeat(1, in_channels // groups, 1, 1, 1)
#     grad_output = grad_output.contiguous().view(
#         grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
#         grad_output.shape[3], grad_output.shape[4])
# 
#     input = input.contiguous().view(1, input.shape[0] * input.shape[1],
#                                     input.shape[2], input.shape[3],
#                                     input.shape[4])
# 
#     grad_weight = torch.conv3d(input, grad_output, None, dilation, padding,
#                                stride, in_channels * min_batch)
# 
#     grad_weight = grad_weight.contiguous().view(
#         min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
#         grad_weight.shape[3], grad_weight.shape[4])
# 
#     return grad_weight.sum(dim=0).view(
#         in_channels // groups, out_channels, grad_weight.shape[2],
#         grad_weight.shape[3], grad_weight.shape[4]).transpose(0, 1).narrow(
#             2, 0, weight_size[2]).narrow(3, 0, weight_size[3]).narrow(
#                 4, 0, weight_size[4])
# 
stop('not implemented')
}

nnf_grid_sample <- function(input, grid, mode = c("bilinear", "nearest"), 
                            padding_mode = c("zeros", "border", "reflection"), 
                            align_corners = FALSE) {
  
  if (mode == "bilinear")
    mode_enum <- 0
  else if (mode == "nearest")
    mode_enum <- 1
  else
    value_error("Unknown mode name '{mode}'. Supported modes are 'bilinear'",
                "and 'nearest'.")
  
  
  if (padding_mode == "zeros")
    padding_mode_enum <- 0
  else if (padding_mode == "border")
    padding_mode_enum <- 1
  else if (padding_mode == "reflection")
    padding_mode_enum <- 2
  else
    value_error("Unknown padding mode name '{padding_mode}'. Supported modes are",
                "'zeros', 'border' and 'reflection'.")
  
  torch_grid_sampler(input = input, grid = grid, interpolation_mode = mode_enum,
                     padding_mode = padding_mode_enum, align_corners = align_corners)
}

nnf_group_norm <- function(input, num_groups, weight = NULL, bias = NULL,
                           eps = 1e-5) {
  
  torch_group_norm(input, input, num_groups = num_groups, weight = weight,
                   bias = bias, eps = eps #TODO ,cudnn_enabled = backends_cudnn_enabled
  )
}

nnf_gumbel_softmax <- function(logits, tau = 1, hard = FALSE, dim = -1) {
  gumbels <- -torch_empty_like(logits, memory_format = torch_contiguous_format())
  gumbels <- gumbels$exponential_()$log()
  gumbels <- (logits + gumbels) / tau
  y_soft <- gumbels$softmax(dim)
  
  if (hard) {
    index <- y_soft$max(dim, keepdim = TRUE)[[2]]
    y_hard <- torch_zeros_like(logits, memory_format = torch_contiguous_format())
    y_hard <- y_hard$scatter_(dim, index, 1)
    ret <- y_hard - y_soft$detach() + y_soft
  } else {
    ret = y_soft
  }
  ret
}

nnf_handle_torch_function <- function() {
# def handle_torch_function(
#         public_api, relevant_args, *args, **kwargs):
#     """Implement a function with checks for __torch_function__ overrides.
# 
#     See torch::autograd::handle_torch_function for the equivalent of this
#     function in the C++ implementation.
# 
#     Arguments
#     ---------
#     public_api : function
#         Function exposed by the public torch API originally called like
#         ``public_api(*args, **kwargs)`` on which arguments are now being
#         checked.
#     relevant_args : iterable
#         Iterable of arguments to check for __torch_function__ methods.
#     args : tuple
#         Arbitrary positional arguments originally passed into ``public_api``.
#     kwargs : tuple
#         Arbitrary keyword arguments originally passed into ``public_api``.
# 
#     Returns
#     -------
#     Result from calling `implementation()` or an `__torch_function__`
#     method, as appropriate.
# 
#     Raises
#     ------
#     TypeError : if no implementation is found.
# 
#     """
#     # Check for __torch_function__ methods.
#     overloaded_args = _get_overloaded_args(relevant_args)
#     # overloaded_args already have unique types.
#     types = tuple(map(type, overloaded_args))
# 
#     # Call overrides
#     for overloaded_arg in overloaded_args:
#         # Use `public_api` instead of `implementation` so __torch_function__
#         # implementations can do equality/identity comparisons.
#         result = overloaded_arg.__torch_function__(public_api, types, args, kwargs)
# 
#         if result is not NotImplemented:
#             return result
# 
#     func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
#     raise TypeError("no implementation found for '{}' on types that implement "
#                     '__torch_function__: {}'
#                     .format(func_name, list(map(type, overloaded_args))))
# 
stop('not implemented')
}

nnf_hardshrink <- function(input, lambd = 0.5) {
  torch_hardshrink(input, lambd)
}

nnf_hardsigmoid <- function(input, inplace = FALSE) {
  if (inplace)
    torch_hardsigmoid_(input)
  else
    torch_hardsigmoid(input)
}

nnf_hardtanh <- function(input, min_val = -1, max_val = 1, inplace = FALSE) {
  if (inplace)
    torch_hardtanh_(input, min_val, max_val)
  else
    torch_hardtanh(input, min_val, max_val)
}

nnf_hardtanh_ <- function(input, min_val = -1, max_val = 1) {
  nnf_hardtanh(input, min_val, max_val, inplace = TRUE)
}

nnf_has_torch_function <- function() {
# def has_torch_function(relevant_args):
#     """Check for __torch_function__ implementations in the elements of an iterable
# 
#     Arguments
#     ---------
#     relevant_args : iterable
#         Iterable or aguments to check for __torch_function__ methods.
# 
#     Returns
#     -------
#     True if any of the elements of relevant_args have __torch_function__
#     implementations, False otherwise.
#     """
#     return any(hasattr(a, '__torch_function__') for a in relevant_args)
# 
stop('not implemented')
}

nnf_hinge_embedding_loss <- function(input, target, margin = 1, reduction = "mean") {
  torch_hinge_embedding_loss(input, target, margin, 
                             reduction = reduction_enum(reduction))
}

nnf_instance_norm <- function(input, running_mean = NULL, running_var = NULL,
                              weight = NULL, bias = NULL, use_input_stats = TRUE,
                              momentum = 0.1, eps = 1e-5) {
  
  torch_instance_norm(input, weight, bias, running_mean, running_var,
                      use_input_stats, momentum, eps, FALSE #TODO backend_cudnn_enabled)
  )
}

interp_output_size <- function(input, size, scale_factor, recompute_scale_factor) {
  dim <- input$dim() - 2
  
  if (is.null(size) && is.null(scale_factor))
    value_error("either size or scale_factor should be defined")
  
  if (!is.null(size) && !is.null(scale_factor))
    value_error("only one of size or scale_factor should be defined")
  
  if (!is.null(scale_factor)) {
    if (length(scale_factor) != dim)
      value_error("scale_factor shape must match input shape.", 
                  "Input is {dim}D, scale_factor size is {lenght(scale_factor)}")
    
  }
  
  if (!is.null(size)) {
    if (length(size) > 1)
      return(size)
    else
      return(rep(size, dim))
  }
  
  if (is.null(scale_factor))
    value_error("assertion failed")
  
  if (length(scale_factor) > 1)
    scale_factors <- scale_factor
  else
    scale_factors <- rep(scale_factor, dim)
  
  
  if (is.null(recompute_scale_factor)) {
    is_float_scale_factor <- FALSE
    
    for (scale in scale_factors) {
      is_float_scale_factor <- floor(scale) != scale
      if (is_float_scale_factor)
        break
    }
    
    warn("The default behavior for interpolate/upsample with float scale_factor will change ",
         "in 1.6.0 to align with other frameworks/libraries, and use scale_factor directly, ",
         "instead of relying on the computed output size. ",
         "If you wish to keep the old behavior, please set recompute_scale_factor=True. ",
         "See the documentation of nn.Upsample for details. ")
  }
  
  
  lapply(
    seq_len(dim),
    function(i) {
      i <- i - 1
      floor(input$size(i + 2) * scale_factors[i + 1])
    }
  )
}

nnf_interpolate <- function(input, size = NULL, scale_factor = NULL, 
                            mode = "nearest", align_corners = FALSE, 
                            recompute_scale_factor = NULL) {
  
  scale_factor_len <- input$dim() - 2
  if (length(scale_factor) == 1)
    scale_factor_repeated <- rep(scale_factor, scale_factor_len)
  else
    scale_factor_repeated <- scale_factor
  
  sfl <- scale_factor_repeated
  sze <- interp_output_size(input, size = size, scale_factor = scale_factor, 
                                    recompute_scale_factor = recompute_scale_factor)
  
  if (input$dim() == 3 && mode == "nearest") {
    return(torch_upsample_nearest1d(input, output_size = sze, scales = sfl[[1]]))
  }
  
  if (input$dim() == 4 && mode == "nearest") {
    return(torch_upsample_nearest2d(input, output_size = sze, scales_h = sfl[[1]],
                                    scales_w = sfl[[2]]))
  }
  
  if (input$dim() == 5 && mode == "nearest") {
    return(torch_upsample_nearest3d(input, sze, sfl[[1]], sfl[[2]], sfl[[3]]))
  }
  
  if (input$dim() == 3 && mode == "area") {
    return(torch_adaptive_avg_pool1d(input, sze))
  }
  
  if (input$dim() == 4 && mode == "area") {
    return(torch_adaptive_avg_pool2d(input, sze))
  }
  
  if (input$dim() == 5 && mode == "area") {
    return(torch_adaptive_avg_pool3d(input, sze))
  }
  
  if (input$dim() == 3 && mode == "linear") {
    return(torch_upsample_linear1d(input, sze, align_corners, sfl[[1]]))
  }
  
  if (input$dim() == 3 && mode == "bilinear") {
    not_implemented_error("Got 3D input, but bilinear mode needs 4D input")
  }
  
  if (input$dim() == 3 && mode == "trilinear") {
    not_implemented_error("Got 3D input, but trilinear mode needs 5D input")
  }
  
  if (input$dim() == 4 && mode == "linear") {
    not_implemented_error("Got 4D input, but trilinear mode needs 3D input")
  }
  
  if (input$dim() == 4 && mode == "bilinear") {
    return(torch_upsample_bilinear2d(input, sze, align_corners, sfl[[1]], sfl[[2]]))
  }
  
  if (input$dim() == 4 && mode == "trilinear") {
    not_implemented_error("Got 4D input, but trilinear mode needs 5D input")
  }
  
  if (input$dim() == 5 && mode == "linear") {
    not_implemented_error("Got 5D input, but trilinear mode needs 3D input")
  }
    
  if (input$dim() == 5 && mode == "bilinear") {
    not_implemented_error("Got 5D input, but bilinear mode needs 4D input")
  }
  
  if (inpt$dim() == 5 && mode == "trilinear") {
    return(torch_upsample_trilinear3d(input, sze, align_corners, sfl[[1]], sfl[[2]], 
                               sfl[[3]]))
  }
  
  if (input$dim() ==4 && mode == "bicubic") {
    return(torch_upsample_bicubic2d(input, sze, align_corners, sfl[[1]], sfl[[2]]))
  }
  
  not_implemented_error("Input Error: Only 3D, 4D and 5D input Tensors supported",
                        " (got {input$dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear",
                        " (got {mode})")
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

nnf_layer_norm <- function() {
# def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
#     # type: (Tensor, List[int], Optional[Tensor], Optional[Tensor], float) -> Tensor
#     r"""Applies Layer Normalization for last certain number of dimensions.
# 
#     See :class:`~torch.nn.LayerNorm` for details.
#     """
#     return torch.layer_norm(input, normalized_shape, weight, bias, eps,
#                             torch.backends.cudnn.enabled)
# 
stop('not implemented')
}

nnf_leaky_relu <- function() {
# def leaky_relu(input, negative_slope=0.01, inplace=False):
#     # type: (Tensor, float, bool) -> Tensor
#     r"""
#     leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor
# 
#     Applies element-wise,
#     :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`
# 
#     See :class:`~torch.nn.LeakyReLU` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 leaky_relu, (input,), input, negative_slope=negative_slope,
#                 inplace=inplace)
#     if inplace:
#         result = torch._C._nn.leaky_relu_(input, negative_slope)
#     else:
#         result = torch._C._nn.leaky_relu(input, negative_slope)
#     return result
# 
stop('not implemented')
}

nnf_leaky_relu_ <- function() {
# 
stop('not implemented')
}

nnf_linear <- function() {
# def linear(input, weight, bias=None):
#     # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
#     r"""
#     Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
# 
#     Shape:
# 
#         - Input: :math:`(N, *, in\_features)` where `*` means any number of
#           additional dimensions
#         - Weight: :math:`(out\_features, in\_features)`
#         - Bias: :math:`(out\_features)`
#         - Output: :math:`(N, *, out\_features)`
#     """
#     tens_ops = (input, weight)
#     if not torch.jit.is_scripting():
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(linear, tens_ops, input, weight, bias=bias)
#     if input.dim() == 2 and bias is not None:
#         # fused op is marginally faster
#         ret = torch.addmm(bias, input, weight.t())
#     else:
#         output = input.matmul(weight.t())
#         if bias is not None:
#             output += bias
#         ret = output
#     return ret
# 
stop('not implemented')
}

nnf_local_response_norm <- function() {
# def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.):
#     # type: (Tensor, int, float, float, float) -> Tensor
#     r"""Applies local response normalization over an input signal composed of
#     several input planes, where channels occupy the second dimension.
#     Applies normalization across channels.
# 
#     See :class:`~torch.nn.LocalResponseNorm` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 local_response_norm, (input,), input, size, alpha=alpha, beta=beta, k=k)
#     dim = input.dim()
#     if dim < 3:
#         raise ValueError('Expected 3D or higher dimensionality \
#                          input (got {} dimensions)'.format(dim))
#     div = input.mul(input).unsqueeze(1)
#     if dim == 3:
#         div = pad(div, (0, 0, size // 2, (size - 1) // 2))
#         div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
#     else:
#         sizes = input.size()
#         div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
#         div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
#         div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
#         div = div.view(sizes)
#     div = div.mul(alpha).add(k).pow(beta)
#     return input / div
# 
stop('not implemented')
}

nnf_log_softmax <- function(input, dim, dtype) {
# def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
#     # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
#     r"""Applies a softmax followed by a logarithm.
# 
#     While mathematically equivalent to log(softmax(x)), doing these two
#     operations separately is slower, and numerically unstable. This function
#     uses an alternative formulation to compute the output and gradient correctly.
# 
#     See :class:`~torch.nn.LogSoftmax` for more details.
# 
#     Arguments:
#         input (Tensor): input
#         dim (int): A dimension along which log_softmax will be computed.
#         dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
#           If specified, the input tensor is casted to :attr:`dtype` before the operation
#           is performed. This is useful for preventing data type overflows. Default: None.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 log_softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
#     if dim is None:
#         dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)
#     if dtype is None:
#         ret = input.log_softmax(dim)
#     else:
#         ret = input.log_softmax(dim, dtype=dtype)
#     return ret
# 
stop('not implemented')
}

nnf_logsigmoid <- function() {
# 
stop('not implemented')
}

nnf_lp_pool1d <- function() {
# def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
#     # type: (Tensor, float, int, Optional[BroadcastingList1[int]], bool) -> Tensor
#     r"""Applies a 1D power-average pooling over an input signal composed of
#     several input planes. If the sum of all inputs to the power of `p` is
#     zero, the gradient is set to zero as well.
# 
#     See :class:`~torch.nn.LPPool1d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 lp_pool1d, (input,), input, norm_type, kernel_size, stride=stride,
#                 ceil_mode=ceil_mode)
#     if stride is not None:
#         out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
#     else:
#         out = avg_pool1d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)
# 
#     return (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1. / norm_type)
# 
stop('not implemented')
}

nnf_lp_pool2d <- function() {
# def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
#     # type: (Tensor, float, int, Optional[BroadcastingList2[int]], bool) -> Tensor
#     r"""Applies a 2D power-average pooling over an input signal composed of
#     several input planes. If the sum of all inputs to the power of `p` is
#     zero, the gradient is set to zero as well.
# 
#     See :class:`~torch.nn.LPPool2d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 lp_pool2d, (input,), input, norm_type, kernel_size, stride=stride,
#                 ceil_mode=ceil_mode)
#     kw, kh = utils._pair(kernel_size)
#     if stride is not None:
#         out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
#     else:
#         out = avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)
# 
#     return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1. / norm_type)
# 
stop('not implemented')
}

nnf_margin_ranking_loss <- function() {
# def margin_ranking_loss(input1, input2, target, margin=0, size_average=None,
#                         reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Tensor, float, Optional[bool], Optional[bool], str) -> Tensor
#     r"""margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor
# 
#     See :class:`~torch.nn.MarginRankingLoss` for details.
#     """  # noqa
#     if not torch.jit.is_scripting():
#         tens_ops = (input1, input2, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 margin_ranking_loss, tens_ops, input1, input2, target, margin=margin,
#                 size_average=size_average, reduce=reduce, reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
#     else:
#         reduction_enum = _Reduction.get_enum(reduction)
#     if input1.dim() == 0 or input2.dim() == 0 or target.dim() == 0:
#         raise RuntimeError(("margin_ranking_loss does not support scalars, got sizes: "
#                             "input1: {}, input2: {}, target: {} ".format(input1.size(), input2.size(), target.size())))
#     return torch.margin_ranking_loss(input1, input2, target, margin, reduction_enum)
# 
stop('not implemented')
}

nnf_math <- function() {
# 
stop('not implemented')
}

nnf_max_pool1d <- function() {
#     def fn(*args, **kwargs):
#         dispatch_flag = False
#         if arg_name in kwargs:
#             dispatch_flag = kwargs[arg_name]
#         elif arg_index < len(args):
#             dispatch_flag = args[arg_index]
# 
#         if dispatch_flag:
#             return if_true(*args, **kwargs)
#         else:
#             return if_false(*args, **kwargs)
# 
stop('not implemented')
}

nnf_max_pool1d_with_indices <- function() {
# def max_pool1d_with_indices(input, kernel_size, stride=None, padding=0,
#                             dilation=1, ceil_mode=False, return_indices=False):
#     # type: (Tensor, BroadcastingList1[int], Optional[BroadcastingList1[int]], BroadcastingList1[int], BroadcastingList1[int], bool, bool) -> Tuple[Tensor, Tensor]  # noqa
#     r"""Applies a 1D max pooling over an input signal composed of several input
#     planes.
# 
#     See :class:`~torch.nn.MaxPool1d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 max_pool1d_with_indices, (input,), input, kernel_size,
#                 stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode,
#                 return_indices=return_indices)
#     if stride is None:
#         stride = torch.jit.annotate(List[int], [])
#     return torch.max_pool1d_with_indices(
#         input, kernel_size, stride, padding, dilation, ceil_mode)
# 
stop('not implemented')
}

nnf_max_pool2d <- function() {
#     def fn(*args, **kwargs):
#         dispatch_flag = False
#         if arg_name in kwargs:
#             dispatch_flag = kwargs[arg_name]
#         elif arg_index < len(args):
#             dispatch_flag = args[arg_index]
# 
#         if dispatch_flag:
#             return if_true(*args, **kwargs)
#         else:
#             return if_false(*args, **kwargs)
# 
stop('not implemented')
}

nnf_max_pool2d_with_indices <- function() {
# def max_pool2d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1,
#                             ceil_mode=False, return_indices=False):
#     # type: (Tensor, BroadcastingList2[int], Optional[BroadcastingList2[int]], BroadcastingList2[int], BroadcastingList2[int], bool, bool) -> Tuple[Tensor, Tensor]  # noqa
#     r"""Applies a 2D max pooling over an input signal composed of several input
#     planes.
# 
#     See :class:`~torch.nn.MaxPool2d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 max_pool2d_with_indices, (input,), input, kernel_size,
#                 stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode,
#                 return_indices=return_indices)
#     if stride is None:
#         stride = torch.jit.annotate(List[int], [])
#     return torch._C._nn.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
# 
stop('not implemented')
}

nnf_max_pool3d <- function() {
#     def fn(*args, **kwargs):
#         dispatch_flag = False
#         if arg_name in kwargs:
#             dispatch_flag = kwargs[arg_name]
#         elif arg_index < len(args):
#             dispatch_flag = args[arg_index]
# 
#         if dispatch_flag:
#             return if_true(*args, **kwargs)
#         else:
#             return if_false(*args, **kwargs)
# 
stop('not implemented')
}

nnf_max_pool3d_with_indices <- function() {
# def max_pool3d_with_indices(input, kernel_size, stride=None, padding=0,
#                             dilation=1, ceil_mode=False, return_indices=False):
#     # type: (Tensor, BroadcastingList3[int], Optional[BroadcastingList3[int]], BroadcastingList3[int], BroadcastingList3[int], bool, bool) -> Tuple[Tensor, Tensor]  # noqa
#     r"""Applies a 3D max pooling over an input signal composed of several input
#     planes.
# 
#     See :class:`~torch.nn.MaxPool3d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 max_pool3d_with_indices, (input,), input, kernel_size,
#                 stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode,
#                 return_indices=return_indices)
#     if stride is None:
#         stride = torch.jit.annotate(List[int], [])
#     return torch._C._nn.max_pool3d_with_indices(
#         input, kernel_size, stride, padding, dilation, ceil_mode)
# 
stop('not implemented')
}

nnf_max_unpool1d <- function() {
# def max_unpool1d(input, indices, kernel_size, stride=None, padding=0,
#                  output_size=None):
#     # type: (Tensor, Tensor, BroadcastingList1[int], Optional[BroadcastingList1[int]], BroadcastingList1[int], Optional[BroadcastingList1[int]]) -> Tensor  # noqa
#     r"""Computes a partial inverse of :class:`MaxPool1d`.
# 
#     See :class:`~torch.nn.MaxUnpool1d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 max_unpool1d, (input,), input, indices, kernel_size,
#                 stride=stride, padding=padding, output_size=output_size)
#     kernel_size = _single(kernel_size)
#     if stride is not None:
#         _stride = _single(stride)
#     else:
#         _stride = kernel_size
#     padding = _single(padding)
#     output_size = _unpool_output_size(input, kernel_size, _stride, padding,
#                                       output_size)
#     if isinstance(output_size, list):
#         output_size = output_size + [1]
#     else:
#         output_size = output_size + (1,)
#     return torch._C._nn.max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3),
#                                      output_size).squeeze(3)
# 
stop('not implemented')
}

nnf_max_unpool2d <- function() {
# def max_unpool2d(input, indices, kernel_size, stride=None, padding=0,
#                  output_size=None):
#     # type: (Tensor, Tensor, BroadcastingList2[int], Optional[BroadcastingList2[int]], BroadcastingList2[int], Optional[BroadcastingList2[int]]) -> Tensor  # noqa
#     r"""Computes a partial inverse of :class:`MaxPool2d`.
# 
#     See :class:`~torch.nn.MaxUnpool2d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 max_unpool2d, (input,), input, indices, kernel_size,
#                 stride=stride, padding=padding, output_size=output_size)
#     kernel_size = _pair(kernel_size)
#     if stride is not None:
#         _stride = _pair(stride)
#     else:
#         _stride = kernel_size
#     padding = _pair(padding)
#     output_size = _unpool_output_size(input, kernel_size, _stride, padding,
#                                       output_size)
#     return torch._C._nn.max_unpool2d(input, indices, output_size)
# 
stop('not implemented')
}

nnf_max_unpool3d <- function() {
# def max_unpool3d(input, indices, kernel_size, stride=None, padding=0,
#                  output_size=None):
#     # type: (Tensor, Tensor, BroadcastingList3[int], Optional[BroadcastingList3[int]], BroadcastingList3[int], Optional[BroadcastingList3[int]]) -> Tensor  # noqa
#     r"""Computes a partial inverse of :class:`MaxPool3d`.
# 
#     See :class:`~torch.nn.MaxUnpool3d` for details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 max_unpool3d, (input,), input, indices, kernel_size,
#                 stride=stride, padding=padding, output_size=output_size)
#     kernel_size = _triple(kernel_size)
#     if stride is not None:
#         _stride = _triple(stride)
#     else:
#         _stride = kernel_size
#     padding = _triple(padding)
#     output_size = _unpool_output_size(input, kernel_size, _stride, padding,
#                                       output_size)
#     return torch._C._nn.max_unpool3d(
#         input, indices, output_size, _stride, padding)
# 
stop('not implemented')
}

nnf_mse_loss <- function() {
# def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
#     r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
# 
#     Measures the element-wise mean squared error.
# 
#     See :class:`~torch.nn.MSELoss` for details.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 mse_loss, tens_ops, input, target, size_average=size_average, reduce=reduce,
#                 reduction=reduction)
#     if not (target.size() == input.size()):
#         warnings.warn("Using a target size ({}) that is different to the input size ({}). "
#                       "This will likely lead to incorrect results due to broadcasting. "
#                       "Please ensure they have the same size.".format(target.size(), input.size()),
#                       stacklevel=2)
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
#     if target.requires_grad:
#         ret = (input - target) ** 2
#         if reduction != 'none':
#             ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
#     else:
#         expanded_input, expanded_target = torch.broadcast_tensors(input, target)
#         ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
#     return ret
# 
stop('not implemented')
}

nnf_multi_head_attention_forward <- function(query, embed_dim_to_check, num_heads, in_proj_weight, bias_k, add_zero_attn, dropout_p, out_proj_weight, training, key_padding_mask, need_weights, attn_mask, use_separate_proj_weight, q_proj_weight) {
# def multi_head_attention_forward(query,                           # type: Tensor
#                                  key,                             # type: Tensor
#                                  value,                           # type: Tensor
#                                  embed_dim_to_check,              # type: int
#                                  num_heads,                       # type: int
#                                  in_proj_weight,                  # type: Tensor
#                                  in_proj_bias,                    # type: Tensor
#                                  bias_k,                          # type: Optional[Tensor]
#                                  bias_v,                          # type: Optional[Tensor]
#                                  add_zero_attn,                   # type: bool
#                                  dropout_p,                       # type: float
#                                  out_proj_weight,                 # type: Tensor
#                                  out_proj_bias,                   # type: Tensor
#                                  training=True,                   # type: bool
#                                  key_padding_mask=None,           # type: Optional[Tensor]
#                                  need_weights=True,               # type: bool
#                                  attn_mask=None,                  # type: Optional[Tensor]
#                                  use_separate_proj_weight=False,  # type: bool
#                                  q_proj_weight=None,              # type: Optional[Tensor]
#                                  k_proj_weight=None,              # type: Optional[Tensor]
#                                  v_proj_weight=None,              # type: Optional[Tensor]
#                                  static_k=None,                   # type: Optional[Tensor]
#                                  static_v=None                    # type: Optional[Tensor]
#                                  ):
#     # type: (...) -> Tuple[Tensor, Optional[Tensor]]
#     r"""
#     Args:
#         query, key, value: map a query and a set of key-value pairs to an output.
#             See "Attention Is All You Need" for more details.
#         embed_dim_to_check: total dimension of the model.
#         num_heads: parallel attention heads.
#         in_proj_weight, in_proj_bias: input projection weight and bias.
#         bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
#         add_zero_attn: add a new batch of zeros to the key and
#                        value sequences at dim=1.
#         dropout_p: probability of an element to be zeroed.
#         out_proj_weight, out_proj_bias: the output projection weight and bias.
#         training: apply dropout if is ``True``.
#         key_padding_mask: if provided, specified padding elements in the key will
#             be ignored by the attention. This is an binary mask. When the value is True,
#             the corresponding value on the attention layer will be filled with -inf.
#         need_weights: output attn_output_weights.
#         attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
#             (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
#             the batches while a 3D mask allows to specify a different mask for the entries of each batch.
#         use_separate_proj_weight: the function accept the proj. weights for query, key,
#             and value in different forms. If false, in_proj_weight will be used, which is
#             a combination of q_proj_weight, k_proj_weight, v_proj_weight.
#         q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
#         static_k, static_v: static key and value used for attention operators.
# 
# 
#     Shape:
#         Inputs:
#         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
#         - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
#           3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
#           S is the source sequence length.
#         - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
#           N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
#         - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
#           N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
# 
#         Outputs:
#         - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
#           E is the embedding dimension.
#         - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
#           L is the target sequence length, S is the source sequence length.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
#                     out_proj_weight, out_proj_bias)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 multi_head_attention_forward, tens_ops, query, key, value,
#                 embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
#                 bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
#                 out_proj_bias, training=training, key_padding_mask=key_padding_mask,
#                 need_weights=need_weights, attn_mask=attn_mask,
#                 use_separate_proj_weight=use_separate_proj_weight,
#                 q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
#                 v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
#     tgt_len, bsz, embed_dim = query.size()
#     assert embed_dim == embed_dim_to_check
#     assert key.size() == value.size()
# 
#     head_dim = embed_dim // num_heads
#     assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#     scaling = float(head_dim) ** -0.5
# 
#     if not use_separate_proj_weight:
#         if torch.equal(query, key) and torch.equal(key, value):
#             # self-attention
#             q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
# 
#         elif torch.equal(key, value):
#             # encoder-decoder attention
#             # This is inline in_proj function with in_proj_weight and in_proj_bias
#             _b = in_proj_bias
#             _start = 0
#             _end = embed_dim
#             _w = in_proj_weight[_start:_end, :]
#             if _b is not None:
#                 _b = _b[_start:_end]
#             q = linear(query, _w, _b)
# 
#             if key is None:
#                 assert value is None
#                 k = None
#                 v = None
#             else:
# 
#                 # This is inline in_proj function with in_proj_weight and in_proj_bias
#                 _b = in_proj_bias
#                 _start = embed_dim
#                 _end = None
#                 _w = in_proj_weight[_start:, :]
#                 if _b is not None:
#                     _b = _b[_start:]
#                 k, v = linear(key, _w, _b).chunk(2, dim=-1)
# 
#         else:
#             # This is inline in_proj function with in_proj_weight and in_proj_bias
#             _b = in_proj_bias
#             _start = 0
#             _end = embed_dim
#             _w = in_proj_weight[_start:_end, :]
#             if _b is not None:
#                 _b = _b[_start:_end]
#             q = linear(query, _w, _b)
# 
#             # This is inline in_proj function with in_proj_weight and in_proj_bias
#             _b = in_proj_bias
#             _start = embed_dim
#             _end = embed_dim * 2
#             _w = in_proj_weight[_start:_end, :]
#             if _b is not None:
#                 _b = _b[_start:_end]
#             k = linear(key, _w, _b)
# 
#             # This is inline in_proj function with in_proj_weight and in_proj_bias
#             _b = in_proj_bias
#             _start = embed_dim * 2
#             _end = None
#             _w = in_proj_weight[_start:, :]
#             if _b is not None:
#                 _b = _b[_start:]
#             v = linear(value, _w, _b)
#     else:
#         q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
#         len1, len2 = q_proj_weight_non_opt.size()
#         assert len1 == embed_dim and len2 == query.size(-1)
# 
#         k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
#         len1, len2 = k_proj_weight_non_opt.size()
#         assert len1 == embed_dim and len2 == key.size(-1)
# 
#         v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
#         len1, len2 = v_proj_weight_non_opt.size()
#         assert len1 == embed_dim and len2 == value.size(-1)
# 
#         if in_proj_bias is not None:
#             q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
#             k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
#             v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
#         else:
#             q = linear(query, q_proj_weight_non_opt, in_proj_bias)
#             k = linear(key, k_proj_weight_non_opt, in_proj_bias)
#             v = linear(value, v_proj_weight_non_opt, in_proj_bias)
#     q = q * scaling
# 
#     if attn_mask is not None:
#         if attn_mask.dim() == 2:
#             attn_mask = attn_mask.unsqueeze(0)
#             if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
#                 raise RuntimeError('The size of the 2D attn_mask is not correct.')
#         elif attn_mask.dim() == 3:
#             if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
#                 raise RuntimeError('The size of the 3D attn_mask is not correct.')
#         else:
#             raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
#         # attn_mask's dim is 3 now.
# 
#     if bias_k is not None and bias_v is not None:
#         if static_k is None and static_v is None:
#             k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
#             v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
#             if attn_mask is not None:
#                 attn_mask = pad(attn_mask, (0, 1))
#             if key_padding_mask is not None:
#                 key_padding_mask = pad(key_padding_mask, (0, 1))
#         else:
#             assert static_k is None, "bias cannot be added to static key."
#             assert static_v is None, "bias cannot be added to static value."
#     else:
#         assert bias_k is None
#         assert bias_v is None
# 
#     q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#     if k is not None:
#         k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#     if v is not None:
#         v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
# 
#     if static_k is not None:
#         assert static_k.size(0) == bsz * num_heads
#         assert static_k.size(2) == head_dim
#         k = static_k
# 
#     if static_v is not None:
#         assert static_v.size(0) == bsz * num_heads
#         assert static_v.size(2) == head_dim
#         v = static_v
# 
#     src_len = k.size(1)
# 
#     if key_padding_mask is not None:
#         assert key_padding_mask.size(0) == bsz
#         assert key_padding_mask.size(1) == src_len
# 
#     if add_zero_attn:
#         src_len += 1
#         k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
#         v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
#         if attn_mask is not None:
#             attn_mask = pad(attn_mask, (0, 1))
#         if key_padding_mask is not None:
#             key_padding_mask = pad(key_padding_mask, (0, 1))
# 
#     attn_output_weights = torch.bmm(q, k.transpose(1, 2))
#     assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
# 
#     if attn_mask is not None:
#         attn_output_weights += attn_mask
# 
#     if key_padding_mask is not None:
#         attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#         attn_output_weights = attn_output_weights.masked_fill(
#             key_padding_mask.unsqueeze(1).unsqueeze(2),
#             float('-inf'),
#         )
#         attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
# 
#     attn_output_weights = softmax(
#         attn_output_weights, dim=-1)
#     attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
# 
#     attn_output = torch.bmm(attn_output_weights, v)
#     assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
#     attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#     attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
# 
#     if need_weights:
#         # average attention weights over heads
#         attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#         return attn_output, attn_output_weights.sum(dim=1) / num_heads
#     else:
#         return attn_output, None
# 
stop('not implemented')
}

nnf_multi_margin_loss <- function() {
# def multi_margin_loss(input, target, p=1, margin=1., weight=None, size_average=None,
#                       reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, int, float, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
#     r"""multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
#                           reduce=None, reduction='mean') -> Tensor
# 
#     See :class:`~torch.nn.MultiMarginLoss` for details.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 multi_margin_loss, tens_ops, input, target, p=p, margin=margin,
#                 weight=weight, size_average=size_average, reduce=reduce,
#                 reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
#     else:
#         reduction_enum = _Reduction.get_enum(reduction)
#     if p != 1 and p != 2:
#         raise ValueError('only p == 1 and p == 2 supported')
#     if weight is not None:
#         if weight.dim() != 1:
#             raise ValueError('weight must be one-dimensional')
# 
#     return torch._C._nn.multi_margin_loss(input, target, p, margin, weight, reduction_enum)
# 
stop('not implemented')
}

nnf_multilabel_margin_loss <- function() {
# def multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
#     r"""multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
# 
#     See :class:`~torch.nn.MultiLabelMarginLoss` for details.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 multilabel_margin_loss, tens_ops, input, target, size_average=size_average,
#                 reduce=reduce, reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
#     else:
#         reduction_enum = _Reduction.get_enum(reduction)
#     return torch._C._nn.multilabel_margin_loss(input, target, reduction_enum)
# 
stop('not implemented')
}

nnf_multilabel_soft_margin_loss <- function() {
# def multilabel_soft_margin_loss(input, target, weight=None, size_average=None,
#                                 reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
#     r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None) -> Tensor
# 
#     See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 multilabel_soft_margin_loss, tens_ops, input, target, weight=weight,
#                 size_average=size_average, reduce=reduce, reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
# 
#     loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))
# 
#     if weight is not None:
#         loss = loss * weight
# 
#     loss = loss.sum(dim=1) / input.size(1)  # only return N loss values
# 
#     if reduction == 'none':
#         ret = loss
#     elif reduction == 'mean':
#         ret = loss.mean()
#     elif reduction == 'sum':
#         ret = loss.sum()
#     else:
#         ret = input
#         raise ValueError(reduction + " is not valid")
#     return ret
# 
stop('not implemented')
}

nnf_nll_loss <- function(input, target, weight, size_average, ignore_index, reduce, reduction) {
# def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
#              reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
#     r"""The negative log likelihood loss.
# 
#     See :class:`~torch.nn.NLLLoss` for details.
# 
#     Args:
#         input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
#             in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
#             in the case of K-dimensional loss.
#         target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
#             or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
#             K-dimensional loss.
#         weight (Tensor, optional): a manual rescaling weight given to each
#             class. If given, has to be a Tensor of size `C`
#         size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
#             the losses are averaged over each loss element in the batch. Note that for
#             some losses, there multiple elements per sample. If the field :attr:`size_average`
#             is set to ``False``, the losses are instead summed for each minibatch. Ignored
#             when reduce is ``False``. Default: ``True``
#         ignore_index (int, optional): Specifies a target value that is ignored
#             and does not contribute to the input gradient. When :attr:`size_average` is
#             ``True``, the loss is averaged over non-ignored targets. Default: -100
#         reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
#             losses are averaged or summed over observations for each minibatch depending
#             on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
#             batch element instead and ignores :attr:`size_average`. Default: ``True``
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
#             ``'mean'``: the sum of the output will be divided by the number of
#             elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
#             and :attr:`reduce` are in the process of being deprecated, and in the meantime,
#             specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
# 
#     Example::
# 
#         >>> # input is of size N x C = 3 x 5
#         >>> input = torch.randn(3, 5, requires_grad=True)
#         >>> # each element in target has to have 0 <= value < C
#         >>> target = torch.tensor([1, 0, 4])
#         >>> output = F.nll_loss(F.log_softmax(input), target)
#         >>> output.backward()
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 nll_loss, tens_ops, input, target, weight=weight, size_average=size_average,
#                 ignore_index=ignore_index, reduce=reduce, reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
#     dim = input.dim()
#     if dim < 2:
#         raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))
# 
#     if input.size(0) != target.size(0):
#         raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
#                          .format(input.size(0), target.size(0)))
#     if dim == 2:
#         ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
#     elif dim == 4:
#         ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
#     else:
#         # dim == 3 or dim > 4
#         n = input.size(0)
#         c = input.size(1)
#         out_size = (n,) + input.size()[2:]
#         if target.size()[1:] != input.size()[2:]:
#             raise ValueError('Expected target size {}, got {}'.format(
#                 out_size, target.size()))
#         input = input.contiguous()
#         target = target.contiguous()
#         # support empty batches, see #15870
#         if input.numel() > 0:
#             input = input.view(n, c, 1, -1)
#         else:
#             input = input.view(n, c, 0, 0)
#         if target.numel() > 0:
#             target = target.view(n, 1, -1)
#         else:
#             target = target.view(n, 0, 0)
#         reduction_enum = _Reduction.get_enum(reduction)
#         if reduction != 'none':
#             ret = torch._C._nn.nll_loss2d(
#                 input, target, weight, reduction_enum, ignore_index)
#         else:
#             out = torch._C._nn.nll_loss2d(
#                 input, target, weight, reduction_enum, ignore_index)
#             ret = out.view(out_size)
#     return ret
# 
stop('not implemented')
}

nnf_normalize <- function(input, p, dim, eps, out) {
# def normalize(input, p=2, dim=1, eps=1e-12, out=None):
#     # type: (Tensor, float, int, float, Optional[Tensor]) -> Tensor
#     r"""Performs :math:`L_p` normalization of inputs over specified dimension.
# 
#     For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
#     :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as
# 
#     .. math::
#         v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
# 
#     With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.
# 
#     Args:
#         input: input tensor of any shape
#         p (float): the exponent value in the norm formulation. Default: 2
#         dim (int): the dimension to reduce. Default: 1
#         eps (float): small value to avoid division by zero. Default: 1e-12
#         out (Tensor, optional): the output tensor. If :attr:`out` is used, this
#                                 operation won't be differentiable.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 normalize, (input,), input, p=p, dim=dim, eps=eps, out=out)
#     if out is None:
#         denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
#         return input / denom
#     else:
#         denom = input.norm(p, dim, keepdim=True).clamp_min_(eps).expand_as(input)
#         return torch.div(input, denom, out=out)
# 
stop('not implemented')
}

nnf_one_hot <- function(tensor, num_classes) {
# 
stop('not implemented')
}

nnf_pad <- function(input, pad, mode, value) {
# def _pad(input, pad, mode='constant', value=0):
#     # type: (Tensor, List[int], str, float) -> Tensor
#     r"""Pads tensor.
# 
#     Padding size:
#         The padding size by which to pad some dimensions of :attr:`input`
#         are described starting from the last dimension and moving forward.
#         :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
#         of ``input`` will be padded.
#         For example, to pad only the last dimension of the input tensor, then
#         :attr:`pad` has the form
#         :math:`(\text{padding\_left}, \text{padding\_right})`;
#         to pad the last 2 dimensions of the input tensor, then use
#         :math:`(\text{padding\_left}, \text{padding\_right},`
#         :math:`\text{padding\_top}, \text{padding\_bottom})`;
#         to pad the last 3 dimensions, use
#         :math:`(\text{padding\_left}, \text{padding\_right},`
#         :math:`\text{padding\_top}, \text{padding\_bottom}`
#         :math:`\text{padding\_front}, \text{padding\_back})`.
# 
#     Padding mode:
#         See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
#         :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
#         padding modes works. Constant padding is implemented for arbitrary dimensions.
#         Replicate padding is implemented for padding the last 3 dimensions of 5D input
#         tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of
#         3D input tensor. Reflect padding is only implemented for padding the last 2
#         dimensions of 4D input tensor, or the last dimension of 3D input tensor.
# 
#     .. include:: cuda_deterministic_backward.rst
# 
#     Args:
#         input (Tensor): N-dimensional tensor
#         pad (tuple): m-elements tuple, where
#             :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
#         mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
#             Default: ``'constant'``
#         value: fill value for ``'constant'`` padding. Default: ``0``
# 
#     Examples::
# 
#         >>> t4d = torch.empty(3, 3, 4, 2)
#         >>> p1d = (1, 1) # pad last dim by 1 on each side
#         >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
#         >>> print(out.size())
#         torch.Size([3, 3, 4, 4])
#         >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
#         >>> out = F.pad(t4d, p2d, "constant", 0)
#         >>> print(out.size())
#         torch.Size([3, 3, 8, 4])
#         >>> t4d = torch.empty(3, 3, 4, 2)
#         >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
#         >>> out = F.pad(t4d, p3d, "constant", 0)
#         >>> print(out.size())
#         torch.Size([3, 9, 7, 3])
# 
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 _pad, (input,), input, pad, mode=mode, value=value)
#     assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
#     assert len(pad) // 2 <= input.dim(), 'Padding length too large'
#     if mode == 'constant':
#         return _VF.constant_pad_nd(input, pad, value)
#     else:
#         assert value == 0, 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
#         if input.dim() == 3:
#             assert len(pad) == 2, '3D tensors expect 2 values for padding'
#             if mode == 'reflect':
#                 return torch._C._nn.reflection_pad1d(input, pad)
#             elif mode == 'replicate':
#                 return torch._C._nn.replication_pad1d(input, pad)
#             elif mode == 'circular':
#                 return _pad_circular(input, pad)
#             else:
#                 raise NotImplementedError
# 
#         elif input.dim() == 4:
#             assert len(pad) == 4, '4D tensors expect 4 values for padding'
#             if mode == 'reflect':
#                 return torch._C._nn.reflection_pad2d(input, pad)
#             elif mode == 'replicate':
#                 return torch._C._nn.replication_pad2d(input, pad)
#             elif mode == 'circular':
#                 return _pad_circular(input, pad)
#             else:
#                 raise NotImplementedError
# 
#         elif input.dim() == 5:
#             assert len(pad) == 6, '5D tensors expect 6 values for padding'
#             if mode == 'reflect':
#                 raise NotImplementedError
#             elif mode == 'replicate':
#                 return torch._C._nn.replication_pad3d(input, pad)
#             elif mode == 'circular':
#                 return _pad_circular(input, pad)
#             else:
#                 raise NotImplementedError
#         else:
#             raise NotImplementedError("Only 3D, 4D, 5D padding with non-constant padding are supported for now")
# 
stop('not implemented')
}

nnf_pairwise_distance <- function() {
# def pairwise_distance(x1, x2, p=2., eps=1e-6, keepdim=False):
#     # type: (Tensor, Tensor, float, float, bool) -> Tensor
#     r"""
#     See :class:`torch.nn.PairwiseDistance` for details
#     """
#     return torch.pairwise_distance(x1, x2, p, eps, keepdim)
# 
stop('not implemented')
}

nnf_pdist <- function(input, p) {
# 
stop('not implemented')
}

nnf_pixel_shuffle <- function(input, upscale_factor) {
# 
stop('not implemented')
}

nnf_poisson_nll_loss <- function(input, target, log_input, full, size_average, eps, reduce, reduction) {
# def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-8,
#                      reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, bool, bool, Optional[bool], float, Optional[bool], str) -> Tensor
#     r"""Poisson negative log likelihood loss.
# 
#     See :class:`~torch.nn.PoissonNLLLoss` for details.
# 
#     Args:
#         input: expectation of underlying Poisson distribution.
#         target: random sample :math:`target \sim \text{Poisson}(input)`.
#         log_input: if ``True`` the loss is computed as
#             :math:`\exp(\text{input}) - \text{target} * \text{input}`, if ``False`` then loss is
#             :math:`\text{input} - \text{target} * \log(\text{input}+\text{eps})`. Default: ``True``
#         full: whether to compute full loss, i. e. to add the Stirling
#             approximation term. Default: ``False``
#             :math:`\text{target} * \log(\text{target}) - \text{target} + 0.5 * \log(2 * \pi * \text{target})`.
#         size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
#             the losses are averaged over each loss element in the batch. Note that for
#             some losses, there multiple elements per sample. If the field :attr:`size_average`
#             is set to ``False``, the losses are instead summed for each minibatch. Ignored
#             when reduce is ``False``. Default: ``True``
#         eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
#             :attr:`log_input`=``False``. Default: 1e-8
#         reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
#             losses are averaged or summed over observations for each minibatch depending
#             on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
#             batch element instead and ignores :attr:`size_average`. Default: ``True``
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
#             ``'mean'``: the sum of the output will be divided by the number of
#             elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
#             and :attr:`reduce` are in the process of being deprecated, and in the meantime,
#             specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
# 
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 poisson_nll_loss, tens_ops, input, target, log_input=log_input, full=full,
#                 size_average=size_average, eps=eps, reduce=reduce, reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
#     if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
#         ret = input
#         raise ValueError(reduction + " is not valid")
# 
#     ret = torch.poisson_nll_loss(input, target, log_input, full, eps, _Reduction.get_enum(reduction))
#     return ret
# 
stop('not implemented')
}

nnf_prelu <- function() {
# def prelu(input, weight):
#     # type: (Tensor, Tensor) -> Tensor
#     r"""prelu(input, weight) -> Tensor
# 
#     Applies element-wise the function
#     :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
#     learnable parameter.
# 
#     See :class:`~torch.nn.PReLU` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(prelu, (input,), input, weight)
#     return torch.prelu(input, weight)
# 
stop('not implemented')
}

nnf_relu <- function() {
# def relu(input, inplace=False):
#     # type: (Tensor, bool) -> Tensor
#     r"""relu(input, inplace=False) -> Tensor
# 
#     Applies the rectified linear unit function element-wise. See
#     :class:`~torch.nn.ReLU` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(relu, (input,), input, inplace=inplace)
#     if inplace:
#         result = torch.relu_(input)
#     else:
#         result = torch.relu(input)
#     return result
# 
stop('not implemented')
}

nnf_relu6 <- function() {
# def relu6(input, inplace=False):
#     # type: (Tensor, bool) -> Tensor
#     r"""relu6(input, inplace=False) -> Tensor
# 
#     Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.
# 
#     See :class:`~torch.nn.ReLU6` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(relu6, (input,), input, inplace=inplace)
#     return hardtanh(input, 0., 6., inplace)
# 
stop('not implemented')
}

nnf_relu_ <- function() {
# 
stop('not implemented')
}

nnf_rrelu <- function() {
# def rrelu(input, lower=1. / 8, upper=1. / 3, training=False, inplace=False):
#     # type: (Tensor, float, float, bool, bool) -> Tensor
#     r"""rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor
# 
#     Randomized leaky ReLU.
# 
#     See :class:`~torch.nn.RReLU` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 rrelu, (input,), input, lower=lower, upper=upper,
#                 training=training, inplace=inplace)
#     if inplace:
#         result = torch.rrelu_(input, lower, upper, training)
#     else:
#         result = torch.rrelu(input, lower, upper, training)
#     return result
# 
stop('not implemented')
}

nnf_rrelu_ <- function() {
# 
stop('not implemented')
}

nnf_selu <- function() {
# def selu(input, inplace=False):
#     # type: (Tensor, bool) -> Tensor
#     r"""selu(input, inplace=False) -> Tensor
# 
#     Applies element-wise,
#     :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`,
#     with :math:`\alpha=1.6732632423543772848170429916717` and
#     :math:`scale=1.0507009873554804934193349852946`.
# 
#     See :class:`~torch.nn.SELU` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(selu, (input,), input, inplace=inplace)
#     if inplace:
#         result = torch.selu_(input)
#     else:
#         result = torch.selu(input)
#     return result
# 
stop('not implemented')
}

nnf_selu_ <- function() {
# 
stop('not implemented')
}

nnf_sigmoid <- function() {
# def sigmoid(input):
#     r"""sigmoid(input) -> Tensor
# 
#     Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`
# 
#     See :class:`~torch.nn.Sigmoid` for more details.
#     """
#     warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
#     return input.sigmoid()
# 
stop('not implemented')
}

nnf_smooth_l1_loss <- function() {
# def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
#     r"""Function that uses a squared term if the absolute
#     element-wise error falls below 1 and an L1 term otherwise.
# 
#     See :class:`~torch.nn.SmoothL1Loss` for details.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 smooth_l1_loss, tens_ops, input, target, size_average=size_average,
#                 reduce=reduce, reduction=reduction)
#     if not (target.size() == input.size()):
#         warnings.warn("Using a target size ({}) that is different to the input size ({}). "
#                       "This will likely lead to incorrect results due to broadcasting. "
#                       "Please ensure they have the same size.".format(target.size(), input.size()),
#                       stacklevel=2)
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
#     if target.requires_grad:
#         ret = _smooth_l1_loss(input, target)
#         if reduction != 'none':
#             ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
#     else:
#         expanded_input, expanded_target = torch.broadcast_tensors(input, target)
#         ret = torch._C._nn.smooth_l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
#     return ret
# 
stop('not implemented')
}

nnf_soft_margin_loss <- function() {
# def soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
#     r"""soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
# 
#     See :class:`~torch.nn.SoftMarginLoss` for details.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 soft_margin_loss, tens_ops, input, target, size_average=size_average,
#                 reduce=reduce, reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
#     else:
#         reduction_enum = _Reduction.get_enum(reduction)
#     return torch._C._nn.soft_margin_loss(input, target, reduction_enum)
# 
stop('not implemented')
}

nnf_softmax <- function(input, dim, dtype) {
# def softmax(input, dim=None, _stacklevel=3, dtype=None):
#     # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
#     r"""Applies a softmax function.
# 
#     Softmax is defined as:
# 
#     :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`
# 
#     It is applied to all slices along dim, and will re-scale them so that the elements
#     lie in the range `[0, 1]` and sum to 1.
# 
#     See :class:`~torch.nn.Softmax` for more details.
# 
#     Arguments:
#         input (Tensor): input
#         dim (int): A dimension along which softmax will be computed.
#         dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
#           If specified, the input tensor is casted to :attr:`dtype` before the operation
#           is performed. This is useful for preventing data type overflows. Default: None.
# 
#     .. note::
#         This function doesn't work directly with NLLLoss,
#         which expects the Log to be computed between the Softmax and itself.
#         Use log_softmax instead (it's faster and has better numerical properties).
# 
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
#     if dim is None:
#         dim = _get_softmax_dim('softmax', input.dim(), _stacklevel)
#     if dtype is None:
#         ret = input.softmax(dim)
#     else:
#         ret = input.softmax(dim, dtype=dtype)
#     return ret
# 
stop('not implemented')
}

nnf_softmin <- function(input, dim, dtype) {
# def softmin(input, dim=None, _stacklevel=3, dtype=None):
#     # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
#     r"""Applies a softmin function.
# 
#     Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`. See softmax definition for mathematical formula.
# 
#     See :class:`~torch.nn.Softmin` for more details.
# 
#     Arguments:
#         input (Tensor): input
#         dim (int): A dimension along which softmin will be computed (so every slice
#             along dim will sum to 1).
#         dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
#           If specified, the input tensor is casted to :attr:`dtype` before the operation
#           is performed. This is useful for preventing data type overflows. Default: None.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 softmin, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
#     if dim is None:
#         dim = _get_softmax_dim('softmin', input.dim(), _stacklevel)
#     if dtype is None:
#         ret = (-input).softmax(dim)
#     else:
#         ret = (-input).softmax(dim, dtype=dtype)
#     return ret
# 
stop('not implemented')
}

nnf_softplus <- function() {
# 
stop('not implemented')
}

nnf_softshrink <- function() {
# 
stop('not implemented')
}

nnf_softsign <- function() {
# def softsign(input):
#     r"""softsign(input) -> Tensor
# 
#     Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`
# 
#     See :class:`~torch.nn.Softsign` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(softsign, (input,), input)
#     return input / (input.abs() + 1)
# 
stop('not implemented')
}

nnf_tanh <- function() {
# def tanh(input):
#     r"""tanh(input) -> Tensor
# 
#     Applies element-wise,
#     :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`
# 
#     See :class:`~torch.nn.Tanh` for more details.
#     """
#     warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
#     return input.tanh()
# 
stop('not implemented')
}

nnf_tanhshrink <- function() {
# def tanhshrink(input):
#     r"""tanhshrink(input) -> Tensor
# 
#     Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`
# 
#     See :class:`~torch.nn.Tanhshrink` for more details.
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(tanhshrink, (input,), input)
#     return input - input.tanh()
# 
stop('not implemented')
}

nnf_threshold <- function() {
# def threshold(input, threshold, value, inplace=False):
#     # type: (Tensor, float, float, bool) -> Tensor
#     r"""Thresholds each element of the input Tensor.
# 
#     See :class:`~torch.nn.Threshold` for more details.
#     """
#     if inplace:
#         result = _VF.threshold_(input, threshold, value)
#     else:
#         result = _VF.threshold(input, threshold, value)
#     return result
# 
stop('not implemented')
}

nnf_threshold_ <- function() {
# 
stop('not implemented')
}

nnf_torch <- function() {
# # @lint-ignore-every PYTHON3COMPATIMPORTS
# 
# r"""
# The torch package contains data structures for multi-dimensional
# tensors and mathematical operations over these are defined.
# Additionally, it provides many utilities for efficient serializing of
# Tensors and arbitrary types, and other useful utilities.
# 
# It has a CUDA counterpart, that enables you to run your tensor computations
# on an NVIDIA GPU with compute capability >= 3.0.
# """
# 
# import os
# import sys
# import platform
# import ctypes
# 
# if sys.version_info < (3,):
#     raise Exception("Python 2 has reached end-of-life and is no longer supported by PyTorch.")
# 
# from ._utils import _import_dotted_name
# from ._utils_internal import get_file_path, prepare_multiprocessing_environment, \
#     USE_RTLD_GLOBAL_WITH_LIBTORCH
# from .version import __version__
# from ._six import string_classes as _string_classes
# 
# __all__ = [
#     'typename', 'is_tensor', 'is_storage', 'set_default_tensor_type',
#     'set_rng_state', 'get_rng_state', 'manual_seed', 'initial_seed', 'seed',
#     'save', 'load', 'set_printoptions', 'chunk', 'split', 'stack', 'matmul',
#     'no_grad', 'enable_grad', 'rand', 'randn',
#     'DoubleStorage', 'FloatStorage', 'LongStorage', 'IntStorage',
#     'ShortStorage', 'CharStorage', 'ByteStorage', 'BoolStorage',
#     'DoubleTensor', 'FloatTensor', 'LongTensor', 'IntTensor',
#     'ShortTensor', 'CharTensor', 'ByteTensor', 'BoolTensor', 'Tensor',
#     'lobpcg',
# ]
# 
# ################################################################################
# # Load the extension module
# ################################################################################
# 
# if platform.system() == 'Windows':
#     is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
#     py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
#     th_dll_path = os.path.join(os.path.dirname(__file__), 'lib')
# 
#     if not os.path.exists(os.path.join(th_dll_path, 'nvToolsExt64_1.dll')) and \
#             not os.path.exists(os.path.join(py_dll_path, 'nvToolsExt64_1.dll')):
#         nvtoolsext_dll_path = os.path.join(
#             os.getenv('NVTOOLSEXT_PATH', 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'), 'bin', 'x64')
#     else:
#         nvtoolsext_dll_path = ''
# 
#     from .version import cuda as cuda_version
#     import glob
#     if cuda_version and len(glob.glob(os.path.join(th_dll_path, 'cudart64*.dll'))) == 0 and \
#             len(glob.glob(os.path.join(py_dll_path, 'cudart64*.dll'))) == 0:
#         cuda_version_1 = cuda_version.replace('.', '_')
#         cuda_path_var = 'CUDA_PATH_V' + cuda_version_1
#         default_path = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v' + cuda_version
#         cuda_path = os.path.join(os.getenv(cuda_path_var, default_path), 'bin')
#     else:
#         cuda_path = ''
# 
#     if sys.version_info >= (3, 8):
#         dll_paths = list(filter(os.path.exists, [th_dll_path, py_dll_path, nvtoolsext_dll_path, cuda_path]))
# 
#         for dll_path in dll_paths:
#             os.add_dll_directory(dll_path)
# 
#     if is_conda or sys.version_info < (3, 8):
#         dll_paths = [th_dll_path, py_dll_path, nvtoolsext_dll_path, cuda_path]
#         dll_paths = list(filter(os.path.exists, dll_paths)) + [os.environ['PATH']]
# 
#         os.environ['PATH'] = ';'.join(dll_paths)
# 
#     import glob
#     dlls = glob.glob(os.path.join(th_dll_path, '*.dll'))
#     for dll in dlls:
#         ctypes.CDLL(dll)
# 
# 
# # See Note [Global dependencies]
# def _load_global_deps():
#     if platform.system() == 'Windows':
#         return
# 
#     lib_name = 'libtorch_global_deps' + ('.dylib' if platform.system() == 'Darwin' else '.so')
#     here = os.path.abspath(__file__)
#     lib_path = os.path.join(os.path.dirname(here), 'lib', lib_name)
# 
#     ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
# 
# 
# if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv('TORCH_USE_RTLD_GLOBAL')) and \
#         platform.system() != 'Windows':
#     # Do it the hard way.  You might want to load libtorch with RTLD_GLOBAL in a
#     # few circumstances:
#     #
#     #   1. You're in a build environment (e.g., fbcode) where
#     #      libtorch_global_deps is not available, but you still need
#     #      to get mkl to link in with RTLD_GLOBAL or it will just
#     #      not work.
#     #
#     #   2. You're trying to run PyTorch under UBSAN and you need
#     #      to ensure that only one copy of libtorch is loaded, so
#     #      vptr checks work properly
#     #
#     # If you're using this setting, you must verify that all the libraries
#     # you load consistently use the same libstdc++, or you may have
#     # mysterious segfaults.
#     #
#     import os as _dl_flags
#     if not hasattr(_dl_flags, 'RTLD_GLOBAL') or not hasattr(_dl_flags, 'RTLD_LAZY'):
#         try:
#             # next try if DLFCN exists
#             import DLFCN as _dl_flags
#         except ImportError:
#             # as a last attempt, use compile-time constants
#             import torch._dl as _dl_flags
#     old_flags = sys.getdlopenflags()
#     sys.setdlopenflags(_dl_flags.RTLD_GLOBAL | _dl_flags.RTLD_LAZY)
#     from torch._C import *
#     sys.setdlopenflags(old_flags)
#     del old_flags
#     del _dl_flags
# 
# else:
#     # Easy way.  You want this most of the time, because it will prevent
#     # C++ symbols from libtorch clobbering C++ symbols from other
#     # libraries, leading to mysterious segfaults.
#     #
#     # See Note [Global dependencies]
#     _load_global_deps()
#     from torch._C import *
# 
# __all__ += [name for name in dir(_C)
#             if name[0] != '_' and
#             not name.endswith('Base')]
# 
# ################################################################################
# # Define basic utilities
# ################################################################################
# 
# 
# def typename(o):
#     if isinstance(o, torch.Tensor):
#         return o.type()
# 
#     module = ''
#     class_name = ''
#     if hasattr(o, '__module__') and o.__module__ != 'builtins' \
#             and o.__module__ != '__builtin__' and o.__module__ is not None:
#         module = o.__module__ + '.'
# 
#     if hasattr(o, '__qualname__'):
#         class_name = o.__qualname__
#     elif hasattr(o, '__name__'):
#         class_name = o.__name__
#     else:
#         class_name = o.__class__.__name__
# 
#     return module + class_name
# 
# 
# def is_tensor(obj):
#     r"""Returns True if `obj` is a PyTorch tensor.
# 
#     Args:
#         obj (Object): Object to test
#     """
#     return isinstance(obj, torch.Tensor)
# 
# 
# def is_storage(obj):
#     r"""Returns True if `obj` is a PyTorch storage object.
# 
#     Args:
#         obj (Object): Object to test
#     """
#     return type(obj) in _storage_classes
# 
# 
# def set_default_tensor_type(t):
#     r"""Sets the default ``torch.Tensor`` type to floating point tensor type
#     ``t``. This type will also be used as default floating point type for
#     type inference in :func:`torch.tensor`.
# 
#     The default floating point tensor type is initially ``torch.FloatTensor``.
# 
#     Args:
#         t (type or string): the floating point tensor type or its name
# 
#     Example::
# 
#         >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
#         torch.float32
#         >>> torch.set_default_tensor_type(torch.DoubleTensor)
#         >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
#         torch.float64
# 
#     """
#     if isinstance(t, _string_classes):
#         t = _import_dotted_name(t)
#     _C._set_default_tensor_type(t)
# 
# 
# def set_default_dtype(d):
#     r"""Sets the default floating point dtype to :attr:`d`. This type will be
#     used as default floating point type for type inference in
#     :func:`torch.tensor`.
# 
#     The default floating point dtype is initially ``torch.float32``.
# 
#     Args:
#         d (:class:`torch.dtype`): the floating point dtype to make the default
# 
#     Example::
# 
#         >>> torch.tensor([1.2, 3]).dtype           # initial default for floating point is torch.float32
#         torch.float32
#         >>> torch.set_default_dtype(torch.float64)
#         >>> torch.tensor([1.2, 3]).dtype           # a new floating point tensor
#         torch.float64
# 
#     """
#     _C._set_default_dtype(d)
# 
# # If you edit these imports, please update torch/__init__.py.in as well
# from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed
# from .serialization import save, load
# from ._tensor_str import set_printoptions
# 
# ################################################################################
# # Define Storage and Tensor classes
# ################################################################################
# 
# from .tensor import Tensor
# from .storage import _StorageBase
# 
# 
# class DoubleStorage(_C.DoubleStorageBase, _StorageBase):
#     pass
# 
# 
# class FloatStorage(_C.FloatStorageBase, _StorageBase):
#     pass
# 
# 
# class HalfStorage(_C.HalfStorageBase, _StorageBase):
#     pass
# 
# 
# class LongStorage(_C.LongStorageBase, _StorageBase):
#     pass
# 
# 
# class IntStorage(_C.IntStorageBase, _StorageBase):
#     pass
# 
# 
# class ShortStorage(_C.ShortStorageBase, _StorageBase):
#     pass
# 
# 
# class CharStorage(_C.CharStorageBase, _StorageBase):
#     pass
# 
# 
# class ByteStorage(_C.ByteStorageBase, _StorageBase):
#     pass
# 
# 
# class BoolStorage(_C.BoolStorageBase, _StorageBase):
#     pass
# 
# 
# class BFloat16Storage(_C.BFloat16StorageBase, _StorageBase):
#     pass
# 
# 
# class QUInt8Storage(_C.QUInt8StorageBase, _StorageBase):
#     pass
# 
# class QInt8Storage(_C.QInt8StorageBase, _StorageBase):
#     pass
# 
# class QInt32Storage(_C.QInt32StorageBase, _StorageBase):
#     pass
# 
# 
# _storage_classes = {
#     DoubleStorage, FloatStorage, LongStorage, IntStorage, ShortStorage,
#     CharStorage, ByteStorage, HalfStorage, BoolStorage, QUInt8Storage, QInt8Storage,
#     QInt32Storage, BFloat16Storage
# }
# 
# # The _tensor_classes set is initialized by the call to _C._initialize_tensor_type_bindings()
# _tensor_classes = set()
# 
# 
# ################################################################################
# # Initialize extension
# ################################################################################
# 
# def manager_path():
#     if platform.system() == 'Windows':
#         return b""
#     path = get_file_path('torch', 'bin', 'torch_shm_manager')
#     prepare_multiprocessing_environment(get_file_path('torch'))
#     if not os.path.exists(path):
#         raise RuntimeError("Unable to find torch_shm_manager at " + path)
#     return path.encode('utf-8')
# 
# 
# # Shared memory manager needs to know the exact location of manager executable
# _C._initExtension(manager_path())
# del manager_path
# 
# for name in dir(_C._VariableFunctions):
#     if name.startswith('__'):
#         continue
#     globals()[name] = getattr(_C._VariableFunctions, name)
# 
# ################################################################################
# # Import interface functions defined in Python
# ################################################################################
# 
# # needs to be after the above ATen bindings so we can overwrite from Python side
# from .functional import *
# 
# 
# ################################################################################
# # Remove unnecessary members
# ################################################################################
# 
# del DoubleStorageBase
# del FloatStorageBase
# del LongStorageBase
# del IntStorageBase
# del ShortStorageBase
# del CharStorageBase
# del ByteStorageBase
# del BoolStorageBase
# del QUInt8StorageBase
# del BFloat16StorageBase
# 
# ################################################################################
# # Import most common subpackages
# ################################################################################
# 
# import torch.cuda
# import torch.autograd
# from torch.autograd import no_grad, enable_grad, set_grad_enabled
# import torch.nn
# import torch.nn.intrinsic
# import torch.nn.quantized
# import torch.optim
# import torch.multiprocessing
# import torch.sparse
# import torch.utils.backcompat
# import torch.onnx
# import torch.jit
# import torch.hub
# import torch.random
# import torch.distributions
# import torch.testing
# import torch.backends.cuda
# import torch.backends.mkl
# import torch.backends.mkldnn
# import torch.backends.openmp
# import torch.backends.quantized
# import torch.quantization
# import torch.utils.data
# import torch.__config__
# import torch.__future__
# 
# _C._init_names(list(torch._storage_classes))
# 
# # attach docstrings to torch and tensor functions
# from . import _torch_docs, _tensor_docs, _storage_docs
# del _torch_docs, _tensor_docs, _storage_docs
# 
# 
# def compiled_with_cxx11_abi():
#     r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
#     return _C._GLIBCXX_USE_CXX11_ABI
# 
# 
# # Import the ops "namespace"
# from torch._ops import ops
# from torch._classes import classes
# 
# # Import the quasi random sampler
# import torch.quasirandom
# 
# # If you are seeing this, it means that this call site was not checked if
# # the memory format could be preserved, and it was switched to old default
# # behaviour of contiguous
# legacy_contiguous_format = contiguous_format
# 
# # Register fork handler to initialize OpenMP in child processes (see gh-28389)
# from torch.multiprocessing._atfork import register_after_fork
# register_after_fork(torch.get_num_threads)
# del register_after_fork
# 
# # Import tools that require fully imported torch (for applying
# # torch.jit.script as a decorator, for instance):
# from ._lobpcg import lobpcg
# 
stop('not implemented')
}

nnf_triplet_margin_loss <- function() {
# def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False, size_average=None,
#                         reduce=None, reduction="mean"):
#     # type: (Tensor, Tensor, Tensor, float, float, float, bool, Optional[bool], Optional[bool], str) -> Tensor
#     r"""
#     See :class:`~torch.nn.TripletMarginLoss` for details
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (anchor, positive, negative)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 triplet_margin_loss, tens_ops, anchor, positive, negative, margin=margin,
#                 p=p, eps=eps, swap=swap, size_average=size_average, reduce=reduce,
#                 reduction=reduction)
#     if size_average is not None or reduce is not None:
#         reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
#     else:
#         reduction_enum = _Reduction.get_enum(reduction)
#     return torch.triplet_margin_loss(anchor, positive, negative, margin, p, eps,
#                                      swap, reduction_enum)
# 
stop('not implemented')
}

nnf_unfold <- function() {
# def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
#     # type: (Tensor, BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int]) -> Tensor  # noqa
#     r"""Extracts sliding local blocks from an batched input tensor.
# 
#     .. warning::
#         Currently, only 4-D input tensors (batched image-like tensors) are
#         supported.
# 
#     .. warning::
# 
#         More than one element of the unfolded tensor may refer to a single
#         memory location. As a result, in-place operations (especially ones that
#         are vectorized) may result in incorrect behavior. If you need to write
#         to the tensor, please clone it first.
# 
# 
#     See :class:`torch.nn.Unfold` for details
#     """
#     if not torch.jit.is_scripting():
#         if type(input) is not Tensor and has_torch_function((input,)):
#             return handle_torch_function(
#                 unfold, (input,), input, kernel_size, dilation=dilation,
#                 padding=padding, stride=stride)
#     if input.dim() == 4:
#         msg = '{} must be int or 2-tuple for 4D input'
#         assert_int_or_pair(kernel_size, 'kernel_size', msg)
#         assert_int_or_pair(dilation, 'dilation', msg)
#         assert_int_or_pair(padding, 'padding', msg)
#         assert_int_or_pair(stride, 'stride', msg)
# 
#         return torch._C._nn.im2col(input, _pair(kernel_size),
#                                    _pair(dilation), _pair(padding), _pair(stride))
#     else:
#         raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))
# 
stop('not implemented')
}

nnf_upsample <- function(input, size, scale_factor, mode, align_corners) {
# def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):  # noqa: F811
#     r"""Upsamples the input to either the given :attr:`size` or the given
#     :attr:`scale_factor`
# 
#     .. warning::
#         This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
#         This is equivalent with ``nn.functional.interpolate(...)``.
# 
#     .. include:: cuda_deterministic_backward.rst
# 
#     The algorithm used for upsampling is determined by :attr:`mode`.
# 
#     Currently temporal, spatial and volumetric upsampling are supported, i.e.
#     expected inputs are 3-D, 4-D or 5-D in shape.
# 
#     The input dimensions are interpreted in the form:
#     `mini-batch x channels x [optional depth] x [optional height] x width`.
# 
#     The modes available for upsampling are: `nearest`, `linear` (3D-only),
#     `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only)
# 
#     Args:
#         input (Tensor): the input tensor
#         size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
#             output spatial size.
#         scale_factor (float or Tuple[float]): multiplier for spatial size. Has to be an integer.
#         mode (string): algorithm used for upsampling:
#             ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
#             ``'trilinear'``. Default: ``'nearest'``
#         align_corners (bool, optional): Geometrically, we consider the pixels of the
#             input and output as squares rather than points.
#             If set to ``True``, the input and output tensors are aligned by the
#             center points of their corner pixels, preserving the values at the corner pixels.
#             If set to ``False``, the input and output tensors are aligned by the corner
#             points of their corner pixels, and the interpolation uses edge value padding
#             for out-of-boundary values, making this operation *independent* of input size
#             when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
#             is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
#             Default: ``False``
# 
#     .. note::
#         With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
#         negative values or values greater than 255 for images.
#         Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
#         when displaying the image.
# 
#     .. warning::
#         With ``align_corners = True``, the linearly interpolating modes
#         (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
#         output and input pixels, and thus the output values can depend on the
#         input size. This was the default behavior for these modes up to version
#         0.3.1. Since then, the default behavior is ``align_corners = False``.
#         See :class:`~torch.nn.Upsample` for concrete examples on how this
#         affects the outputs.
# 
#     """
#     warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
#     return interpolate(input, size, scale_factor, mode, align_corners)
# 
stop('not implemented')
}

nnf_upsample_bilinear <- function(input, size, scale_factor) {
# def upsample_bilinear(input, size=None, scale_factor=None):  # noqa: F811
#     r"""Upsamples the input, using bilinear upsampling.
# 
#     .. warning::
#         This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
#         This is equivalent with
#         ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.
# 
#     Expected inputs are spatial (4 dimensional). Use `upsample_trilinear` fo
#     volumetric (5 dimensional) inputs.
# 
#     Args:
#         input (Tensor): input
#         size (int or Tuple[int, int]): output spatial size.
#         scale_factor (int or Tuple[int, int]): multiplier for spatial size
# 
#     .. include:: cuda_deterministic_backward.rst
#     """
#     # DeprecationWarning is ignored by default
#     warnings.warn("nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.")
#     return interpolate(input, size, scale_factor, mode='bilinear', align_corners=True)
# 
stop('not implemented')
}

nnf_upsample_nearest <- function(input, size, scale_factor) {
# def upsample_nearest(input, size=None, scale_factor=None):  # noqa: F811
#     r"""Upsamples the input, using nearest neighbours' pixel values.
# 
#     .. warning::
#         This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
#         This is equivalent with ``nn.functional.interpolate(..., mode='nearest')``.
# 
#     Currently spatial and volumetric upsampling are supported (i.e. expected
#     inputs are 4 or 5 dimensional).
# 
#     Args:
#         input (Tensor): input
#         size (int or Tuple[int, int] or Tuple[int, int, int]): output spatia
#             size.
#         scale_factor (int): multiplier for spatial size. Has to be an integer.
# 
#     .. include:: cuda_deterministic_backward.rst
#     """
#     # DeprecationWarning is ignored by default
#     warnings.warn("nn.functional.upsample_nearest is deprecated. Use nn.functional.interpolate instead.")
#     return interpolate(input, size, scale_factor, mode='nearest')
# 
stop('not implemented')
}

nnf_warnings <- function() {
# """Python part of the warnings subsystem."""
# 
# import sys
# 
# 
# __all__ = ["warn", "warn_explicit", "showwarning",
#            "formatwarning", "filterwarnings", "simplefilter",
#            "resetwarnings", "catch_warnings"]
# 
# def showwarning(message, category, filename, lineno, file=None, line=None):
#     """Hook to write a warning to a file; replace if you like."""
#     msg = WarningMessage(message, category, filename, lineno, file, line)
#     _showwarnmsg_impl(msg)
# 
# def formatwarning(message, category, filename, lineno, line=None):
#     """Function to format a warning the standard way."""
#     msg = WarningMessage(message, category, filename, lineno, None, line)
#     return _formatwarnmsg_impl(msg)
# 
# def _showwarnmsg_impl(msg):
#     file = msg.file
#     if file is None:
#         file = sys.stderr
#         if file is None:
#             # sys.stderr is None when run with pythonw.exe:
#             # warnings get lost
#             return
#     text = _formatwarnmsg(msg)
#     try:
#         file.write(text)
#     except OSError:
#         # the file (probably stderr) is invalid - this warning gets lost.
#         pass
# 
# def _formatwarnmsg_impl(msg):
#     category = msg.category.__name__
#     s =  f"{msg.filename}:{msg.lineno}: {category}: {msg.message}\n"
# 
#     if msg.line is None:
#         try:
#             import linecache
#             line = linecache.getline(msg.filename, msg.lineno)
#         except Exception:
#             # When a warning is logged during Python shutdown, linecache
#             # and the import machinery don't work anymore
#             line = None
#             linecache = None
#     else:
#         line = msg.line
#     if line:
#         line = line.strip()
#         s += "  %s\n" % line
# 
#     if msg.source is not None:
#         try:
#             import tracemalloc
#         # Logging a warning should not raise a new exception:
#         # catch Exception, not only ImportError and RecursionError.
#         except Exception:
#             # don't suggest to enable tracemalloc if it's not available
#             tracing = True
#             tb = None
#         else:
#             tracing = tracemalloc.is_tracing()
#             try:
#                 tb = tracemalloc.get_object_traceback(msg.source)
#             except Exception:
#                 # When a warning is logged during Python shutdown, tracemalloc
#                 # and the import machinery don't work anymore
#                 tb = None
# 
#         if tb is not None:
#             s += 'Object allocated at (most recent call last):\n'
#             for frame in tb:
#                 s += ('  File "%s", lineno %s\n'
#                       % (frame.filename, frame.lineno))
# 
#                 try:
#                     if linecache is not None:
#                         line = linecache.getline(frame.filename, frame.lineno)
#                     else:
#                         line = None
#                 except Exception:
#                     line = None
#                 if line:
#                     line = line.strip()
#                     s += '    %s\n' % line
#         elif not tracing:
#             s += (f'{category}: Enable tracemalloc to get the object '
#                   f'allocation traceback\n')
#     return s
# 
# # Keep a reference to check if the function was replaced
# _showwarning_orig = showwarning
# 
# def _showwarnmsg(msg):
#     """Hook to write a warning to a file; replace if you like."""
#     try:
#         sw = showwarning
#     except NameError:
#         pass
#     else:
#         if sw is not _showwarning_orig:
#             # warnings.showwarning() was replaced
#             if not callable(sw):
#                 raise TypeError("warnings.showwarning() must be set to a "
#                                 "function or method")
# 
#             sw(msg.message, msg.category, msg.filename, msg.lineno,
#                msg.file, msg.line)
#             return
#     _showwarnmsg_impl(msg)
# 
# # Keep a reference to check if the function was replaced
# _formatwarning_orig = formatwarning
# 
# def _formatwarnmsg(msg):
#     """Function to format a warning the standard way."""
#     try:
#         fw = formatwarning
#     except NameError:
#         pass
#     else:
#         if fw is not _formatwarning_orig:
#             # warnings.formatwarning() was replaced
#             return fw(msg.message, msg.category,
#                       msg.filename, msg.lineno, msg.line)
#     return _formatwarnmsg_impl(msg)
# 
# def filterwarnings(action, message="", category=Warning, module="", lineno=0,
#                    append=False):
#     """Insert an entry into the list of warnings filters (at the front).
# 
#     'action' -- one of "error", "ignore", "always", "default", "module",
#                 or "once"
#     'message' -- a regex that the warning message must match
#     'category' -- a class that the warning must be a subclass of
#     'module' -- a regex that the module name must match
#     'lineno' -- an integer line number, 0 matches all warnings
#     'append' -- if true, append to the list of filters
#     """
#     assert action in ("error", "ignore", "always", "default", "module",
#                       "once"), "invalid action: %r" % (action,)
#     assert isinstance(message, str), "message must be a string"
#     assert isinstance(category, type), "category must be a class"
#     assert issubclass(category, Warning), "category must be a Warning subclass"
#     assert isinstance(module, str), "module must be a string"
#     assert isinstance(lineno, int) and lineno >= 0, \
#            "lineno must be an int >= 0"
# 
#     if message or module:
#         import re
# 
#     if message:
#         message = re.compile(message, re.I)
#     else:
#         message = None
#     if module:
#         module = re.compile(module)
#     else:
#         module = None
# 
#     _add_filter(action, message, category, module, lineno, append=append)
# 
# def simplefilter(action, category=Warning, lineno=0, append=False):
#     """Insert a simple entry into the list of warnings filters (at the front).
# 
#     A simple filter matches all modules and messages.
#     'action' -- one of "error", "ignore", "always", "default", "module",
#                 or "once"
#     'category' -- a class that the warning must be a subclass of
#     'lineno' -- an integer line number, 0 matches all warnings
#     'append' -- if true, append to the list of filters
#     """
#     assert action in ("error", "ignore", "always", "default", "module",
#                       "once"), "invalid action: %r" % (action,)
#     assert isinstance(lineno, int) and lineno >= 0, \
#            "lineno must be an int >= 0"
#     _add_filter(action, None, category, None, lineno, append=append)
# 
# def _add_filter(*item, append):
#     # Remove possible duplicate filters, so new one will be placed
#     # in correct place. If append=True and duplicate exists, do nothing.
#     if not append:
#         try:
#             filters.remove(item)
#         except ValueError:
#             pass
#         filters.insert(0, item)
#     else:
#         if item not in filters:
#             filters.append(item)
#     _filters_mutated()
# 
# def resetwarnings():
#     """Clear the list of warning filters, so that no filters are active."""
#     filters[:] = []
#     _filters_mutated()
# 
# class _OptionError(Exception):
#     """Exception used by option processing helpers."""
#     pass
# 
# # Helper to process -W options passed via sys.warnoptions
# def _processoptions(args):
#     for arg in args:
#         try:
#             _setoption(arg)
#         except _OptionError as msg:
#             print("Invalid -W option ignored:", msg, file=sys.stderr)
# 
# # Helper for _processoptions()
# def _setoption(arg):
#     parts = arg.split(':')
#     if len(parts) > 5:
#         raise _OptionError("too many fields (max 5): %r" % (arg,))
#     while len(parts) < 5:
#         parts.append('')
#     action, message, category, module, lineno = [s.strip()
#                                                  for s in parts]
#     action = _getaction(action)
#     category = _getcategory(category)
#     if message or module:
#         import re
#     if message:
#         message = re.escape(message)
#     if module:
#         module = re.escape(module) + r'\Z'
#     if lineno:
#         try:
#             lineno = int(lineno)
#             if lineno < 0:
#                 raise ValueError
#         except (ValueError, OverflowError):
#             raise _OptionError("invalid lineno %r" % (lineno,)) from None
#     else:
#         lineno = 0
#     filterwarnings(action, message, category, module, lineno)
# 
# # Helper for _setoption()
# def _getaction(action):
#     if not action:
#         return "default"
#     if action == "all": return "always" # Alias
#     for a in ('default', 'always', 'ignore', 'module', 'once', 'error'):
#         if a.startswith(action):
#             return a
#     raise _OptionError("invalid action: %r" % (action,))
# 
# # Helper for _setoption()
# def _getcategory(category):
#     if not category:
#         return Warning
#     if '.' not in category:
#         import builtins as m
#         klass = category
#     else:
#         module, _, klass = category.rpartition('.')
#         try:
#             m = __import__(module, None, None, [klass])
#         except ImportError:
#             raise _OptionError("invalid module name: %r" % (module,)) from None
#     try:
#         cat = getattr(m, klass)
#     except AttributeError:
#         raise _OptionError("unknown warning category: %r" % (category,)) from None
#     if not issubclass(cat, Warning):
#         raise _OptionError("invalid warning category: %r" % (category,))
#     return cat
# 
# 
# def _is_internal_frame(frame):
#     """Signal whether the frame is an internal CPython implementation detail."""
#     filename = frame.f_code.co_filename
#     return 'importlib' in filename and '_bootstrap' in filename
# 
# 
# def _next_external_frame(frame):
#     """Find the next frame that doesn't involve CPython internals."""
#     frame = frame.f_back
#     while frame is not None and _is_internal_frame(frame):
#         frame = frame.f_back
#     return frame
# 
# 
# # Code typically replaced by _warnings
# def warn(message, category=None, stacklevel=1, source=None):
#     """Issue a warning, or maybe ignore it or raise an exception."""
#     # Check if message is already a Warning object
#     if isinstance(message, Warning):
#         category = message.__class__
#     # Check category argument
#     if category is None:
#         category = UserWarning
#     if not (isinstance(category, type) and issubclass(category, Warning)):
#         raise TypeError("category must be a Warning subclass, "
#                         "not '{:s}'".format(type(category).__name__))
#     # Get context information
#     try:
#         if stacklevel <= 1 or _is_internal_frame(sys._getframe(1)):
#             # If frame is too small to care or if the warning originated in
#             # internal code, then do not try to hide any frames.
#             frame = sys._getframe(stacklevel)
#         else:
#             frame = sys._getframe(1)
#             # Look for one frame less since the above line starts us off.
#             for x in range(stacklevel-1):
#                 frame = _next_external_frame(frame)
#                 if frame is None:
#                     raise ValueError
#     except ValueError:
#         globals = sys.__dict__
#         lineno = 1
#     else:
#         globals = frame.f_globals
#         lineno = frame.f_lineno
#     if '__name__' in globals:
#         module = globals['__name__']
#     else:
#         module = "<string>"
#     filename = globals.get('__file__')
#     if filename:
#         fnl = filename.lower()
#         if fnl.endswith(".pyc"):
#             filename = filename[:-1]
#     else:
#         if module == "__main__":
#             try:
#                 filename = sys.argv[0]
#             except AttributeError:
#                 # embedded interpreters don't have sys.argv, see bug #839151
#                 filename = '__main__'
#         if not filename:
#             filename = module
#     registry = globals.setdefault("__warningregistry__", {})
#     warn_explicit(message, category, filename, lineno, module, registry,
#                   globals, source)
# 
# def warn_explicit(message, category, filename, lineno,
#                   module=None, registry=None, module_globals=None,
#                   source=None):
#     lineno = int(lineno)
#     if module is None:
#         module = filename or "<unknown>"
#         if module[-3:].lower() == ".py":
#             module = module[:-3] # XXX What about leading pathname?
#     if registry is None:
#         registry = {}
#     if registry.get('version', 0) != _filters_version:
#         registry.clear()
#         registry['version'] = _filters_version
#     if isinstance(message, Warning):
#         text = str(message)
#         category = message.__class__
#     else:
#         text = message
#         message = category(message)
#     key = (text, category, lineno)
#     # Quick test for common case
#     if registry.get(key):
#         return
#     # Search the filters
#     for item in filters:
#         action, msg, cat, mod, ln = item
#         if ((msg is None or msg.match(text)) and
#             issubclass(category, cat) and
#             (mod is None or mod.match(module)) and
#             (ln == 0 or lineno == ln)):
#             break
#     else:
#         action = defaultaction
#     # Early exit actions
#     if action == "ignore":
#         return
# 
#     # Prime the linecache for formatting, in case the
#     # "file" is actually in a zipfile or something.
#     import linecache
#     linecache.getlines(filename, module_globals)
# 
#     if action == "error":
#         raise message
#     # Other actions
#     if action == "once":
#         registry[key] = 1
#         oncekey = (text, category)
#         if onceregistry.get(oncekey):
#             return
#         onceregistry[oncekey] = 1
#     elif action == "always":
#         pass
#     elif action == "module":
#         registry[key] = 1
#         altkey = (text, category, 0)
#         if registry.get(altkey):
#             return
#         registry[altkey] = 1
#     elif action == "default":
#         registry[key] = 1
#     else:
#         # Unrecognized actions are errors
#         raise RuntimeError(
#               "Unrecognized action (%r) in warnings.filters:\n %s" %
#               (action, item))
#     # Print message and context
#     msg = WarningMessage(message, category, filename, lineno, source)
#     _showwarnmsg(msg)
# 
# 
# class WarningMessage(object):
# 
#     _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file",
#                         "line", "source")
# 
#     def __init__(self, message, category, filename, lineno, file=None,
#                  line=None, source=None):
#         self.message = message
#         self.category = category
#         self.filename = filename
#         self.lineno = lineno
#         self.file = file
#         self.line = line
#         self.source = source
#         self._category_name = category.__name__ if category else None
# 
#     def __str__(self):
#         return ("{message : %r, category : %r, filename : %r, lineno : %s, "
#                     "line : %r}" % (self.message, self._category_name,
#                                     self.filename, self.lineno, self.line))
# 
# 
# class catch_warnings(object):
# 
#     """A context manager that copies and restores the warnings filter upon
#     exiting the context.
# 
#     The 'record' argument specifies whether warnings should be captured by a
#     custom implementation of warnings.showwarning() and be appended to a list
#     returned by the context manager. Otherwise None is returned by the context
#     manager. The objects appended to the list are arguments whose attributes
#     mirror the arguments to showwarning().
# 
#     The 'module' argument is to specify an alternative module to the module
#     named 'warnings' and imported under that name. This argument is only useful
#     when testing the warnings module itself.
# 
#     """
# 
#     def __init__(self, *, record=False, module=None):
#         """Specify whether to record warnings and if an alternative module
#         should be used other than sys.modules['warnings'].
# 
#         For compatibility with Python 3.0, please consider all arguments to be
#         keyword-only.
# 
#         """
#         self._record = record
#         self._module = sys.modules['warnings'] if module is None else module
#         self._entered = False
# 
#     def __repr__(self):
#         args = []
#         if self._record:
#             args.append("record=True")
#         if self._module is not sys.modules['warnings']:
#             args.append("module=%r" % self._module)
#         name = type(self).__name__
#         return "%s(%s)" % (name, ", ".join(args))
# 
#     def __enter__(self):
#         if self._entered:
#             raise RuntimeError("Cannot enter %r twice" % self)
#         self._entered = True
#         self._filters = self._module.filters
#         self._module.filters = self._filters[:]
#         self._module._filters_mutated()
#         self._showwarning = self._module.showwarning
#         self._showwarnmsg_impl = self._module._showwarnmsg_impl
#         if self._record:
#             log = []
#             self._module._showwarnmsg_impl = log.append
#             # Reset showwarning() to the default implementation to make sure
#             # that _showwarnmsg() calls _showwarnmsg_impl()
#             self._module.showwarning = self._module._showwarning_orig
#             return log
#         else:
#             return None
# 
#     def __exit__(self, *exc_info):
#         if not self._entered:
#             raise RuntimeError("Cannot exit %r without entering first" % self)
#         self._module.filters = self._filters
#         self._module._filters_mutated()
#         self._module.showwarning = self._showwarning
#         self._module._showwarnmsg_impl = self._showwarnmsg_impl
# 
# 
# # Private utility function called by _PyErr_WarnUnawaitedCoroutine
# def _warn_unawaited_coroutine(coro):
#     msg_lines = [
#         f"coroutine '{coro.__qualname__}' was never awaited\n"
#     ]
#     if coro.cr_origin is not None:
#         import linecache, traceback
#         def extract():
#             for filename, lineno, funcname in reversed(coro.cr_origin):
#                 line = linecache.getline(filename, lineno)
#                 yield (filename, lineno, funcname, line)
#         msg_lines.append("Coroutine created at (most recent call last)\n")
#         msg_lines += traceback.format_list(list(extract()))
#     msg = "".join(msg_lines).rstrip("\n")
#     # Passing source= here means that if the user happens to have tracemalloc
#     # enabled and tracking where the coroutine was created, the warning will
#     # contain that traceback. This does mean that if they have *both*
#     # coroutine origin tracking *and* tracemalloc enabled, they'll get two
#     # partially-redundant tracebacks. If we wanted to be clever we could
#     # probably detect this case and avoid it, but for now we don't bother.
#     warn(msg, category=RuntimeWarning, stacklevel=2, source=coro)
# 
# 
# # filters contains a sequence of filter 5-tuples
# # The components of the 5-tuple are:
# # - an action: error, ignore, always, default, module, or once
# # - a compiled regex that must match the warning message
# # - a class representing the warning category
# # - a compiled regex that must match the module that is being warned
# # - a line number for the line being warning, or 0 to mean any line
# # If either if the compiled regexs are None, match anything.
# try:
#     from _warnings import (filters, _defaultaction, _onceregistry,
#                            warn, warn_explicit, _filters_mutated)
#     defaultaction = _defaultaction
#     onceregistry = _onceregistry
#     _warnings_defaults = True
# except ImportError:
#     filters = []
#     defaultaction = "default"
#     onceregistry = {}
# 
#     _filters_version = 1
# 
#     def _filters_mutated():
#         global _filters_version
#         _filters_version += 1
# 
#     _warnings_defaults = False
# 
# 
# # Module initialization
# _processoptions(sys.warnoptions)
# if not _warnings_defaults:
#     # Several warning categories are ignored by default in regular builds
#     if not hasattr(sys, 'gettotalrefcount'):
#         filterwarnings("default", category=DeprecationWarning,
#                        module="__main__", append=1)
#         simplefilter("ignore", category=DeprecationWarning, append=1)
#         simplefilter("ignore", category=PendingDeprecationWarning, append=1)
#         simplefilter("ignore", category=ImportWarning, append=1)
#         simplefilter("ignore", category=ResourceWarning, append=1)
# 
# del _warnings_defaults
# 
stop('not implemented')
}