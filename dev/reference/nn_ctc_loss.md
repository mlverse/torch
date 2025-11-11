# The Connectionist Temporal Classification loss.

Calculates loss between a continuous (unsegmented) time series and a
target sequence. CTCLoss sums over the probability of possible
alignments of input to target, producing a loss value which is
differentiable with respect to each input node. The alignment of input
to target is assumed to be "many-to-one", which limits the length of the
target sequence such that it must be \\\leq\\ the input length.

## Usage

``` r
nn_ctc_loss(blank = 0, reduction = "mean", zero_infinity = FALSE)
```

## Arguments

- blank:

  (int, optional): blank label. Default \\0\\.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the output losses will be divided by the target
  lengths and then the mean over the batch is taken. Default: `'mean'`

- zero_infinity:

  (bool, optional): Whether to zero infinite losses and the associated
  gradients. Default: `FALSE` Infinite losses mainly occur when the
  inputs are too short to be aligned to the targets.

## Note

In order to use CuDNN, the following must be satisfied: `targets` must
be in concatenated format, all `input_lengths` must be `T`. \\blank=0\\,
`target_lengths` \\\leq 256\\, the integer arguments must be of The
regular implementation uses the (more common in PyTorch) `torch_long`
dtype. dtype `torch_int32`.

In some circumstances when using the CUDA backend with CuDNN, this
operator may select a nondeterministic algorithm to increase
performance. If this is undesirable, you can try to make the operation
deterministic (potentially at a performance cost) by setting
`torch.backends.cudnn.deterministic = TRUE`.

## Shape

- Log_probs: Tensor of size \\(T, N, C)\\, where \\T = \mbox{input
  length}\\, \\N = \mbox{batch size}\\, and \\C = \mbox{number of
  classes (including blank)}\\. The logarithmized probabilities of the
  outputs (e.g. obtained with \[nnf)log_softmax()\]).

- Targets: Tensor of size \\(N, S)\\ or
  \\(\mbox{sum}(\mbox{target\\lengths}))\\, where \\N = \mbox{batch
  size}\\ and \\S = \mbox{max target length, if shape is } (N, S)\\. It
  represent the target sequences. Each element in the target sequence is
  a class index. And the target index cannot be blank (default=0). In
  the \\(N, S)\\ form, targets are padded to the length of the longest
  sequence, and stacked. In the \\(\mbox{sum}(\mbox{target\\lengths}))\\
  form, the targets are assumed to be un-padded and concatenated within
  1 dimension.

- Input_lengths: Tuple or tensor of size \\(N)\\, where \\N =
  \mbox{batch size}\\. It represent the lengths of the inputs (must each
  be \\\leq T\\). And the lengths are specified for each sequence to
  achieve masking under the assumption that sequences are padded to
  equal lengths.

- Target_lengths: Tuple or tensor of size \\(N)\\, where \\N =
  \mbox{batch size}\\. It represent lengths of the targets. Lengths are
  specified for each sequence to achieve masking under the assumption
  that sequences are padded to equal lengths. If target shape is
  \\(N,S)\\, target_lengths are effectively the stop index \\s_n\\ for
  each target sequence, such that `target_n = targets[n,0:s_n]` for each
  target in a batch. Lengths must each be \\\leq S\\ If the targets are
  given as a 1d tensor that is the concatenation of individual targets,
  the target_lengths must add up to the total length of the tensor.

- Output: scalar. If `reduction` is `'none'`, then \\(N)\\, where \\N =
  \mbox{batch size}\\.

\[nnf)log_softmax()\]: R:nnf)log_softmax() \[n,0:s_n\]: R:n,0:s_n

## References

A. Graves et al.: Connectionist Temporal Classification: Labelling
Unsegmented Sequence Data with Recurrent Neural Networks:
https://www.cs.toronto.edu/~graves/icml_2006.pdf

## Examples

``` r
if (torch_is_installed()) {
# Target are to be padded
T <- 50 # Input sequence length
C <- 20 # Number of classes (including blank)
N <- 16 # Batch size
S <- 30 # Target sequence length of longest target in batch (padding length)
S_min <- 10 # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input <- torch_randn(T, N, C)$log_softmax(2)$detach()$requires_grad_()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target <- torch_randint(low = 1, high = C, size = c(N, S), dtype = torch_long())

input_lengths <- torch_full(size = c(N), fill_value = TRUE, dtype = torch_long())
target_lengths <- torch_randint(low = S_min, high = S, size = c(N), dtype = torch_long())
ctc_loss <- nn_ctc_loss()
loss <- ctc_loss(input, target, input_lengths, target_lengths)
loss$backward()


# Target are to be un-padded
T <- 50 # Input sequence length
C <- 20 # Number of classes (including blank)
N <- 16 # Batch size

# Initialize random batch of input vectors, for *size = (T,N,C)
input <- torch_randn(T, N, C)$log_softmax(2)$detach()$requires_grad_()
input_lengths <- torch_full(size = c(N), fill_value = TRUE, dtype = torch_long())

# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths <- torch_randint(low = 1, high = T, size = c(N), dtype = torch_long())
target <- torch_randint(
  low = 1, high = C, size = as.integer(sum(target_lengths)),
  dtype = torch_long()
)
ctc_loss <- nn_ctc_loss()
loss <- ctc_loss(input, target, input_lengths, target_lengths)
loss$backward()
}
```
