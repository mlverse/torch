# Embedding module

A simple lookup table that stores embeddings of a fixed dictionary and
size. This module is often used to store word embeddings and retrieve
them using indices. The input to the module is a list of indices, and
the output is the corresponding word embeddings.

## Usage

``` r
nn_embedding(
  num_embeddings,
  embedding_dim,
  padding_idx = NULL,
  max_norm = NULL,
  norm_type = 2,
  scale_grad_by_freq = FALSE,
  sparse = FALSE,
  .weight = NULL
)
```

## Arguments

- num_embeddings:

  (int): size of the dictionary of embeddings

- embedding_dim:

  (int): the size of each embedding vector

- padding_idx:

  (int, optional): If given, pads the output with the embedding vector
  at `padding_idx` (initialized to zeros) whenever it encounters the
  index.

- max_norm:

  (float, optional): If given, each embedding vector with norm larger
  than `max_norm` is renormalized to have norm `max_norm`.

- norm_type:

  (float, optional): The p of the p-norm to compute for the `max_norm`
  option. Default `2`.

- scale_grad_by_freq:

  (boolean, optional): If given, this will scale gradients by the
  inverse of frequency of the words in the mini-batch. Default `False`.

- sparse:

  (bool, optional): If `True`, gradient w.r.t. `weight` matrix will be a
  sparse tensor.

- .weight:

  (Tensor) embeddings weights (in case you want to set it manually)

  See Notes for more details regarding sparse gradients.

## Note

Keep in mind that only a limited number of optimizers support sparse
gradients: currently it's `optim.SGD` (`CUDA` and `CPU`),
`optim.SparseAdam` (`CUDA` and `CPU`) and `optim.Adagrad` (`CPU`)

With `padding_idx` set, the embedding vector at `padding_idx` is
initialized to all zeros. However, note that this vector can be modified
afterwards, e.g., using a customized initialization method, and thus
changing the vector used to pad the output. The gradient for this vector
from nn_embedding is always zero.

## Attributes

- weight (Tensor): the learnable weights of the module of shape
  (num_embeddings, embedding_dim) initialized from \\\mathcal{N}(0, 1)\\

## Shape

- Input: \\(\*)\\, LongTensor of arbitrary shape containing the indices
  to extract

- Output: \\(\*, H)\\, where `*` is the input shape and
  \\H=\mbox{embedding\\dim}\\

## Examples

``` r
if (torch_is_installed()) {
# an Embedding module containing 10 tensors of size 3
embedding <- nn_embedding(10, 3)
# a batch of 2 samples of 4 indices each
input <- torch_tensor(rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)), dtype = torch_long())
embedding(input)
# example with padding_idx
embedding <- nn_embedding(10, 3, padding_idx = 1)
input <- torch_tensor(matrix(c(1, 3, 1, 6), nrow = 1), dtype = torch_long())
embedding(input)
}
#> torch_tensor
#> (1,.,.) = 
#>   0.0000  0.0000  0.0000
#>  -1.5285 -1.1152 -0.8753
#>   0.0000  0.0000  0.0000
#>   1.3611 -0.2303 -1.0041
#> [ CPUFloatType{1,4,3} ][ grad_fn = <EmbeddingBackward0> ]
```
