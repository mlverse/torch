

# -> adaptive_avg_pool1d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_avg_pool1d
#'
#' @examples
#'
#' 
NULL
# -> adaptive_avg_pool1d <-

# -> adaptive_avg_pool2d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_avg_pool2d
#'
#' @examples
#'
#' 
NULL
# -> adaptive_avg_pool2d <-

# -> adaptive_avg_pool3d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_avg_pool3d
#'
#' @examples
#'
#' 
NULL
# -> adaptive_avg_pool3d <-

# -> adaptive_max_pool1d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_max_pool1d
#'
#' @examples
#'
#' 
NULL
# -> adaptive_max_pool1d <-

# -> adaptive_max_pool1d_with_indices: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_max_pool1d_with_indices
#'
#' @examples
#'
#' 
NULL
# -> adaptive_max_pool1d_with_indices <-

# -> adaptive_max_pool2d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_max_pool2d
#'
#' @examples
#'
#' 
NULL
# -> adaptive_max_pool2d <-

# -> adaptive_max_pool2d_with_indices: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_max_pool2d_with_indices
#'
#' @examples
#'
#' 
NULL
# -> adaptive_max_pool2d_with_indices <-

# -> adaptive_max_pool3d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_max_pool3d
#'
#' @examples
#'
#' 
NULL
# -> adaptive_max_pool3d <-

# -> adaptive_max_pool3d_with_indices: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_adaptive_max_pool3d_with_indices
#'
#' @examples
#'
#' 
NULL
# -> adaptive_max_pool3d_with_indices <-

# -> affine_grid: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_affine_grid
#'
#' @examples
#'
#' 
NULL
# -> affine_grid <-

# -> alpha_dropout: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_alpha_dropout
#'
#' @examples
#'
#' 
NULL
# -> alpha_dropout <-

# -> avg_pool1d: 728fbca0128db2740f4d9f241cb971e1 <-
#'
#' @name nnf_avg_pool1d
#'
#' @examples
#'
#' # pool of square window of size=3, stride=2
#' input = torch_tensor(c([[1, 2, 3, 4, 5, 6, 7]]), dtype=torch_float32())
#' F$avg_pool1d(input, kernel_size=3, stride=2)
NULL
# -> avg_pool1d <-

# -> avg_pool2d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_avg_pool2d
#'
#' @examples
#'
#' 
NULL
# -> avg_pool2d <-

# -> avg_pool3d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_avg_pool3d
#'
#' @examples
#'
#' 
NULL
# -> avg_pool3d <-

# -> batch_norm: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_batch_norm
#'
#' @examples
#'
#' 
NULL
# -> batch_norm <-

# -> bilinear: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_bilinear
#'
#' @examples
#'
#' 
NULL
# -> bilinear <-

# -> binary_cross_entropy: 70a8c5e2680566239c0a8b12f3016935 <-
#'
#' @name nnf_binary_cross_entropy
#'
#' @examples
#'
#' input = torch_randn(list(3, 2), requires_grad=TRUE)
#' target = torch_rand(list(3, 2), requires_grad=FALSE)
#' loss = F$binary_cross_entropy(F$sigmoid(input), target)
#' loss$backward()
NULL
# -> binary_cross_entropy <-

# -> binary_cross_entropy_with_logits: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_binary_cross_entropy_with_logits
#'
#' @examples
#'
#' 
NULL
# -> binary_cross_entropy_with_logits <-

# -> boolean_dispatch: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_boolean_dispatch
#'
#' @examples
#'
#' 
NULL
# -> boolean_dispatch <-

# -> celu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_celu
#'
#' @examples
#'
#' 
NULL
# -> celu <-

# -> celu_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_celu_
#'
#' @examples
#'
#' 
NULL
# -> celu_ <-

# -> conv1d: 3a9090d974ee61599fba1c1c9f3e1d14 <-
#'
#' @name nnf_conv1d
#'
#' @examples
#'
#' filters = torch_randn(c(33, 16, 3))
#' inputs = torch_randn(c(20, 16, 50))
#' F$conv1d(inputs, filters)
NULL
# -> conv1d <-

# -> conv2d: 73bd1e08715f8f25547a70fb821a6eea <-
#'
#' @name nnf_conv2d
#'
#' @examples
#'
#' # With square kernels and equal stride
#' filters = torch_randn(c(8,4,3,3))
#' inputs = torch_randn(c(1,4,5,5))
#' F$conv2d(inputs, filters, padding=1)
NULL
# -> conv2d <-

# -> conv3d: f644490bb91b9559df9913760cd99203 <-
#'
#' @name nnf_conv3d
#'
#' @examples
#'
#' filters = torch_randn(c(33, 16, 3, 3, 3))
#' inputs = torch_randn(c(20, 16, 50, 10, 20))
#' F$conv3d(inputs, filters)
NULL
# -> conv3d <-

# -> conv_tbc: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_conv_tbc
#'
#' @examples
#'
#' 
NULL
# -> conv_tbc <-

# -> conv_transpose1d: 69815461ca09463f23c426d0529d9611 <-
#'
#' @name nnf_conv_transpose1d
#'
#' @examples
#'
#' inputs = torch_randn(c(20, 16, 50))
#' weights = torch_randn(c(16, 33, 5))
#' F$conv_transpose1d(inputs, weights)
NULL
# -> conv_transpose1d <-

# -> conv_transpose2d: 599038ae972c64325923c50172c1c083 <-
#'
#' @name nnf_conv_transpose2d
#'
#' @examples
#'
#' # With square kernels and equal stride
#' inputs = torch_randn(c(1, 4, 5, 5))
#' weights = torch_randn(c(4, 8, 3, 3))
#' F$conv_transpose2d(inputs, weights, padding=1)
NULL
# -> conv_transpose2d <-

# -> conv_transpose3d: 4699ff7e852ff50d9a34e456612f461d <-
#'
#' @name nnf_conv_transpose3d
#'
#' @examples
#'
#' inputs = torch_randn(c(20, 16, 50, 10, 20))
#' weights = torch_randn(c(16, 33, 3, 3, 3))
#' F$conv_transpose3d(inputs, weights)
NULL
# -> conv_transpose3d <-

# -> cosine_embedding_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_cosine_embedding_loss
#'
#' @examples
#'
#' 
NULL
# -> cosine_embedding_loss <-

# -> cosine_similarity: f6f66e6ba78bca3dfdeca3b8fd941f36 <-
#'
#' @name nnf_cosine_similarity
#'
#' @examples
#'
#' input1 = torch_randn(c(100, 128))
#' input2 = torch_randn(c(100, 128))
#' output = F$cosine_similarity(input1, input2)
#' print(output)
NULL
# -> cosine_similarity <-

# -> cross_entropy: 8524344283dad4e5f11b494e3ba63b01 <-
#'
#' @name nnf_cross_entropy
#'
#' @examples
#'
#' input = torch_randn(3, 5, requires_grad=TRUE)
#' target = torch_randint(5, list(3,), dtype=torch_int64())
#' loss = F$cross_entropy(input, target)
#' loss$backward()
NULL
# -> cross_entropy <-

# -> ctc_loss: 79dae0d58930665f5b0f1408c2fd43a2 <-
#'
#' @name nnf_ctc_loss
#'
#' @examples
#'
#' log_probs = torch_randn(c(50, 16, 20))$log_softmax(2)$detach()$requires_grad_()
#' targets = torch_randint(1, 20, list(16, 30), dtype=torch_long())
#' input_lengths = torch_full(list(16,), 50, dtype=torch_long())
#' target_lengths = torch_randint(10,30,list(16,), dtype=torch_long())
#' loss = F$ctc_loss(log_probs, targets, input_lengths, target_lengths)
#' loss$backward()
NULL
# -> ctc_loss <-

# -> dropout: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_dropout
#'
#' @examples
#'
#' 
NULL
# -> dropout <-

# -> dropout2d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_dropout2d
#'
#' @examples
#'
#' 
NULL
# -> dropout2d <-

# -> dropout3d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_dropout3d
#'
#' @examples
#'
#' 
NULL
# -> dropout3d <-

# -> elu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_elu
#'
#' @examples
#'
#' 
NULL
# -> elu <-

# -> elu_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_elu_
#'
#' @examples
#'
#' 
NULL
# -> elu_ <-

# -> embedding: 0a01941200ba3e17e43e030a8e535ac9 <-
#'
#' @name nnf_embedding
#'
#' @examples
#'
#' # a batch of 2 samples of 4 indices each
#' input = torch_tensor(c([1,2,4,5],[4,3,2,9]))
#' # an embedding matrix containing 10 tensors of size 3
#' embedding_matrix = torch_rand(10, 3)
#' F$embedding(input, embedding_matrix)
#' # example with padding_idx
#' weights = torch_rand(10, 3)
#' weightsc(0, :)$zero_()
#' embedding_matrix = weights
#' input = torch_tensor(c([0,2,0,5]))
#' F$embedding(input, embedding_matrix, padding_idx=0)
NULL
# -> embedding <-

# -> embedding_bag: f314abf49988966327853652192304ff <-
#'
#' @name nnf_embedding_bag
#'
#' @examples
#'
#' # an Embedding module containing 10 tensors of size 3
#' embedding_matrix = torch_rand(10, 3)
#' # a batch of 2 samples of 4 indices each
#' input = torch_tensor(c(1,2,4,5,4,3,2,9))
#' offsets = torch_tensor(c(0,4))
#' F$embedding_bag(embedding_matrix, input, offsets)
NULL
# -> embedding_bag <-

# -> fold: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_fold
#'
#' @examples
#'
#' 
NULL
# -> fold <-

# -> fractional_max_pool2d: 3e328ebb822b85f1df6f79705f0e1818 <-
#'
#' @name nnf_fractional_max_pool2d
#'
#' @examples
#'
#' input = torch_randn(c(20, 16, 50, 32))
#' # pool of square window of size=3, and target output size 13x12
#' F$fractional_max_pool2d(input, 3, output_size=list(13, 12))
#' # pool of square window and target output size being half of input image size
#' F$fractional_max_pool2d(input, 3, output_ratio=list(0.5, 0.5))
NULL
# -> fractional_max_pool2d <-

# -> fractional_max_pool2d_with_indices: 3e328ebb822b85f1df6f79705f0e1818 <-
#'
#' @name nnf_fractional_max_pool2d_with_indices
#'
#' @examples
#'
#' input = torch_randn(c(20, 16, 50, 32))
#' # pool of square window of size=3, and target output size 13x12
#' F$fractional_max_pool2d(input, 3, output_size=list(13, 12))
#' # pool of square window and target output size being half of input image size
#' F$fractional_max_pool2d(input, 3, output_ratio=list(0.5, 0.5))
NULL
# -> fractional_max_pool2d_with_indices <-

# -> fractional_max_pool3d: ce81a313b7f347b8d5bed0a2ec3627cc <-
#'
#' @name nnf_fractional_max_pool3d
#'
#' @examples
#'
#' input = torch_randn(c(20, 16, 50, 32, 16))
#' # pool of cubic window of size=3, and target output size 13x12x11
#' F$fractional_max_pool3d(input, 3, output_size=list(13, 12, 11))
#' # pool of cubic window and target output size being half of input size
#' F$fractional_max_pool3d(input, 3, output_ratio=list(0.5, 0.5, 0.5))
NULL
# -> fractional_max_pool3d <-

# -> fractional_max_pool3d_with_indices: ce81a313b7f347b8d5bed0a2ec3627cc <-
#'
#' @name nnf_fractional_max_pool3d_with_indices
#'
#' @examples
#'
#' input = torch_randn(c(20, 16, 50, 32, 16))
#' # pool of cubic window of size=3, and target output size 13x12x11
#' F$fractional_max_pool3d(input, 3, output_size=list(13, 12, 11))
#' # pool of cubic window and target output size being half of input size
#' F$fractional_max_pool3d(input, 3, output_ratio=list(0.5, 0.5, 0.5))
NULL
# -> fractional_max_pool3d_with_indices <-

# -> gelu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_gelu
#'
#' @examples
#'
#' 
NULL
# -> gelu <-

# -> glu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_glu
#'
#' @examples
#'
#' 
NULL
# -> glu <-

# -> grad: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_grad
#'
#' @examples
#'
#' 
NULL
# -> grad <-

# -> grid_sample: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_grid_sample
#'
#' @examples
#'
#' 
NULL
# -> grid_sample <-

# -> group_norm: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_group_norm
#'
#' @examples
#'
#' 
NULL
# -> group_norm <-

# -> gumbel_softmax: 1513938d5ede21019e7467bac166398a <-
#'
#' @name nnf_gumbel_softmax
#'
#' @examples
#'
#' logits = torch_randn(c(20, 32))
#' # Sample soft categorical using reparametrization trick:
#' F$gumbel_softmax(logits, tau=1, hard=FALSE)
#' # Sample hard categorical using "Straight-through" trick:
#' F$gumbel_softmax(logits, tau=1, hard=TRUE)
NULL
# -> gumbel_softmax <-

# -> handle_torch_function: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_handle_torch_function
#'
#' @examples
#'
#' 
NULL
# -> handle_torch_function <-

# -> hardshrink: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_hardshrink
#'
#' @examples
#'
#' 
NULL
# -> hardshrink <-

# -> hardsigmoid: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_hardsigmoid
#'
#' @examples
#'
#' 
NULL
# -> hardsigmoid <-

# -> hardtanh: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_hardtanh
#'
#' @examples
#'
#' 
NULL
# -> hardtanh <-

# -> hardtanh_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_hardtanh_
#'
#' @examples
#'
#' 
NULL
# -> hardtanh_ <-

# -> has_torch_function: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_has_torch_function
#'
#' @examples
#'
#' 
NULL
# -> has_torch_function <-

# -> hinge_embedding_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_hinge_embedding_loss
#'
#' @examples
#'
#' 
NULL
# -> hinge_embedding_loss <-

# -> instance_norm: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_instance_norm
#'
#' @examples
#'
#' 
NULL
# -> instance_norm <-

# -> interpolate: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_interpolate
#'
#' @examples
#'
#' 
NULL
# -> interpolate <-

# -> kl_div: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_kl_div
#'
#' @examples
#'
#' 
NULL
# -> kl_div <-

# -> l1_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_l1_loss
#'
#' @examples
#'
#' 
NULL
# -> l1_loss <-

# -> layer_norm: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_layer_norm
#'
#' @examples
#'
#' 
NULL
# -> layer_norm <-

# -> leaky_relu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_leaky_relu
#'
#' @examples
#'
#' 
NULL
# -> leaky_relu <-

# -> leaky_relu_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_leaky_relu_
#'
#' @examples
#'
#' 
NULL
# -> leaky_relu_ <-

# -> linear: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_linear
#'
#' @examples
#'
#' 
NULL
# -> linear <-

# -> local_response_norm: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_local_response_norm
#'
#' @examples
#'
#' 
NULL
# -> local_response_norm <-

# -> log_softmax: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_log_softmax
#'
#' @examples
#'
#' 
NULL
# -> log_softmax <-

# -> logsigmoid: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_logsigmoid
#'
#' @examples
#'
#' 
NULL
# -> logsigmoid <-

# -> lp_pool1d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_lp_pool1d
#'
#' @examples
#'
#' 
NULL
# -> lp_pool1d <-

# -> lp_pool2d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_lp_pool2d
#'
#' @examples
#'
#' 
NULL
# -> lp_pool2d <-

# -> margin_ranking_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_margin_ranking_loss
#'
#' @examples
#'
#' 
NULL
# -> margin_ranking_loss <-

# -> math: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_math
#'
#' @examples
#'
#' 
NULL
# -> math <-

# -> max_pool1d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_pool1d
#'
#' @examples
#'
#' 
NULL
# -> max_pool1d <-

# -> max_pool1d_with_indices: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_pool1d_with_indices
#'
#' @examples
#'
#' 
NULL
# -> max_pool1d_with_indices <-

# -> max_pool2d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_pool2d
#'
#' @examples
#'
#' 
NULL
# -> max_pool2d <-

# -> max_pool2d_with_indices: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_pool2d_with_indices
#'
#' @examples
#'
#' 
NULL
# -> max_pool2d_with_indices <-

# -> max_pool3d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_pool3d
#'
#' @examples
#'
#' 
NULL
# -> max_pool3d <-

# -> max_pool3d_with_indices: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_pool3d_with_indices
#'
#' @examples
#'
#' 
NULL
# -> max_pool3d_with_indices <-

# -> max_unpool1d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_unpool1d
#'
#' @examples
#'
#' 
NULL
# -> max_unpool1d <-

# -> max_unpool2d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_unpool2d
#'
#' @examples
#'
#' 
NULL
# -> max_unpool2d <-

# -> max_unpool3d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_max_unpool3d
#'
#' @examples
#'
#' 
NULL
# -> max_unpool3d <-

# -> mse_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_mse_loss
#'
#' @examples
#'
#' 
NULL
# -> mse_loss <-

# -> multi_head_attention_forward: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_multi_head_attention_forward
#'
#' @examples
#'
#' 
NULL
# -> multi_head_attention_forward <-

# -> multi_margin_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_multi_margin_loss
#'
#' @examples
#'
#' 
NULL
# -> multi_margin_loss <-

# -> multilabel_margin_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_multilabel_margin_loss
#'
#' @examples
#'
#' 
NULL
# -> multilabel_margin_loss <-

# -> multilabel_soft_margin_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_multilabel_soft_margin_loss
#'
#' @examples
#'
#' 
NULL
# -> multilabel_soft_margin_loss <-

# -> nll_loss: 0c598d5b8e502455d7749b77825faf32 <-
#'
#' @name nnf_nll_loss
#'
#' @examples
#'
#' # input is of size N x C = 3 x 5
#' input = torch_randn(3, 5, requires_grad=TRUE)
#' # each element in target has to have 0 <= value < C
#' target = torch_tensor(c(1, 0, 4))
#' output = F$nll_loss(F$log_softmax(input), target)
#' output$backward()
NULL
# -> nll_loss <-

# -> normalize: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_normalize
#'
#' @examples
#'
#' 
NULL
# -> normalize <-

# -> one_hot: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_one_hot
#'
#' @examples
#'
#' 
NULL
# -> one_hot <-

# -> pad: c2abad03d4b92c337770bf559aa2b59d <-
#'
#' @name nnf_pad
#'
#' @examples
#'
#' t4d = torch_empty(3, 3, 4, 2)
#' p1d = list(1, 1) # pad last dim by 1 on each side
#' out = F$pad(t4d, p1d, "constant", 0)  # effectively zero padding
#' print(out$size())
#' p2d = list(1, 1, 2, 2) # pad last dim by list(1, 1) and 2nd to last by list(2, 2)
#' out = F$pad(t4d, p2d, "constant", 0)
#' print(out$size())
#' t4d = torch_empty(3, 3, 4, 2)
#' p3d = list(0, 1, 2, 1, 3, 3) # pad by list(0, 1), list(2, 1), and list(3, 3)
#' out = F$pad(t4d, p3d, "constant", 0)
#' print(out$size())
NULL
# -> pad <-

# -> pairwise_distance: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_pairwise_distance
#'
#' @examples
#'
#' 
NULL
# -> pairwise_distance <-

# -> pdist: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_pdist
#'
#' @examples
#'
#' 
NULL
# -> pdist <-

# -> pixel_shuffle: a768693934e6bd930deee49f0153701d <-
#'
#' @name nnf_pixel_shuffle
#'
#' @examples
#'
#' input = torch_randn(c(1, 9, 4, 4))
#' output = torch_nn$functional$pixel_shuffle(input, 3)
#' print(output$size())
NULL
# -> pixel_shuffle <-

# -> poisson_nll_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_poisson_nll_loss
#'
#' @examples
#'
#' 
NULL
# -> poisson_nll_loss <-

# -> prelu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_prelu
#'
#' @examples
#'
#' 
NULL
# -> prelu <-

# -> relu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_relu
#'
#' @examples
#'
#' 
NULL
# -> relu <-

# -> relu6: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_relu6
#'
#' @examples
#'
#' 
NULL
# -> relu6 <-

# -> relu_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_relu_
#'
#' @examples
#'
#' 
NULL
# -> relu_ <-

# -> rrelu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_rrelu
#'
#' @examples
#'
#' 
NULL
# -> rrelu <-

# -> rrelu_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_rrelu_
#'
#' @examples
#'
#' 
NULL
# -> rrelu_ <-

# -> selu: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_selu
#'
#' @examples
#'
#' 
NULL
# -> selu <-

# -> selu_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_selu_
#'
#' @examples
#'
#' 
NULL
# -> selu_ <-

# -> sigmoid: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_sigmoid
#'
#' @examples
#'
#' 
NULL
# -> sigmoid <-

# -> smooth_l1_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_smooth_l1_loss
#'
#' @examples
#'
#' 
NULL
# -> smooth_l1_loss <-

# -> soft_margin_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_soft_margin_loss
#'
#' @examples
#'
#' 
NULL
# -> soft_margin_loss <-

# -> softmax: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_softmax
#'
#' @examples
#'
#' 
NULL
# -> softmax <-

# -> softmin: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_softmin
#'
#' @examples
#'
#' 
NULL
# -> softmin <-

# -> softplus: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_softplus
#'
#' @examples
#'
#' 
NULL
# -> softplus <-

# -> softshrink: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_softshrink
#'
#' @examples
#'
#' 
NULL
# -> softshrink <-

# -> softsign: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_softsign
#'
#' @examples
#'
#' 
NULL
# -> softsign <-

# -> tanh: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_tanh
#'
#' @examples
#'
#' 
NULL
# -> tanh <-

# -> tanhshrink: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_tanhshrink
#'
#' @examples
#'
#' 
NULL
# -> tanhshrink <-

# -> threshold: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_threshold
#'
#' @examples
#'
#' 
NULL
# -> threshold <-

# -> threshold_: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_threshold_
#'
#' @examples
#'
#' 
NULL
# -> threshold_ <-

# -> torch: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_torch
#'
#' @examples
#'
#' 
NULL
# -> torch <-

# -> triplet_margin_loss: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_triplet_margin_loss
#'
#' @examples
#'
#' 
NULL
# -> triplet_margin_loss <-

# -> unfold: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_unfold
#'
#' @examples
#'
#' 
NULL
# -> unfold <-

# -> upsample: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_upsample
#'
#' @examples
#'
#' 
NULL
# -> upsample <-

# -> upsample_bilinear: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_upsample_bilinear
#'
#' @examples
#'
#' 
NULL
# -> upsample_bilinear <-

# -> upsample_nearest: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_upsample_nearest
#'
#' @examples
#'
#' 
NULL
# -> upsample_nearest <-

# -> warnings: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name nnf_warnings
#'
#' @examples
#'
#' 
NULL
# -> warnings <-