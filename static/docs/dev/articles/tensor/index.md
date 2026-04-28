# Tensor objects

``` r
library(torch)
```

Central to torch is the `torch_tensor` objects. `torch_tensor`’s are R
objects very similar to R6 instances. Tensors have a large amount of
methods that can be called using the `$` operator.

Following is a list of all methods that can be called by tensor objects
and their documentation. You can also look at [PyTorch’s
documentation](https://docs.pytorch.org/docs/stable/tensors.html) for
additional details.

## numpy_T

Is this Tensor with its dimensions reversed.

If `n` is the number of dimensions in `x`, `x$numpy_T()` is equivalent
to `x$permute(n, n-1, ..., 1)`.

## abs

abs() -\> Tensor

See
[`?torch_abs`](https://torch.mlverse.org/docs/dev/reference/torch_abs.md)

## abs\_

abs\_() -\> Tensor

In-place version of `$abs`

## absolute

absolute() -\> Tensor

Alias for \[\$abs()\]

## absolute\_

absolute\_() -\> Tensor

In-place version of `$absolute` Alias for \[\$abs\_()\]

## acos

acos() -\> Tensor

See
[`?torch_acos`](https://torch.mlverse.org/docs/dev/reference/torch_acos.md)

## acos\_

acos\_() -\> Tensor

In-place version of `$acos`

## acosh

acosh() -\> Tensor

See
[`?torch_acosh`](https://torch.mlverse.org/docs/dev/reference/torch_acosh.md)

## acosh\_

acosh\_() -\> Tensor

In-place version of `$acosh`

## add

add(other, \*, alpha=1) -\> Tensor

Add a scalar or tensor to `self` tensor. If both `alpha` and `other` are
specified, each element of `other` is scaled by `alpha` before being
used.

When `other` is a tensor, the shape of `other` must be broadcastable
with the shape of the underlying tensor

See
[`?torch_add`](https://torch.mlverse.org/docs/dev/reference/torch_add.md)

## add\_

add\_(other, \*, alpha=1) -\> Tensor

In-place version of `$add`

## addbmm

addbmm(batch1, batch2, \*, beta=1, alpha=1) -\> Tensor

See
[`?torch_addbmm`](https://torch.mlverse.org/docs/dev/reference/torch_addbmm.md)

## addbmm\_

addbmm\_(batch1, batch2, \*, beta=1, alpha=1) -\> Tensor

In-place version of `$addbmm`

## addcdiv

addcdiv(tensor1, tensor2, \*, value=1) -\> Tensor

See
[`?torch_addcdiv`](https://torch.mlverse.org/docs/dev/reference/torch_addcdiv.md)

## addcdiv\_

addcdiv\_(tensor1, tensor2, \*, value=1) -\> Tensor

In-place version of `$addcdiv`

## addcmul

addcmul(tensor1, tensor2, \*, value=1) -\> Tensor

See
[`?torch_addcmul`](https://torch.mlverse.org/docs/dev/reference/torch_addcmul.md)

## addcmul\_

addcmul\_(tensor1, tensor2, \*, value=1) -\> Tensor

In-place version of `$addcmul`

## addmm

addmm(mat1, mat2, \*, beta=1, alpha=1) -\> Tensor

See
[`?torch_addmm`](https://torch.mlverse.org/docs/dev/reference/torch_addmm.md)

## addmm\_

addmm\_(mat1, mat2, \*, beta=1, alpha=1) -\> Tensor

In-place version of `$addmm`

## addmv

addmv(mat, vec, \*, beta=1, alpha=1) -\> Tensor

See
[`?torch_addmv`](https://torch.mlverse.org/docs/dev/reference/torch_addmv.md)

## addmv\_

addmv\_(mat, vec, \*, beta=1, alpha=1) -\> Tensor

In-place version of `$addmv`

## addr

addr(vec1, vec2, \*, beta=1, alpha=1) -\> Tensor

See
[`?torch_addr`](https://torch.mlverse.org/docs/dev/reference/torch_addr.md)

## addr\_

addr\_(vec1, vec2, \*, beta=1, alpha=1) -\> Tensor

In-place version of `$addr`

## align_as

align_as(other) -\> Tensor

Permutes the dimensions of the `self` tensor to match the dimension
order in the `other` tensor, adding size-one dims for any new names.

This operation is useful for explicit broadcasting by names (see
examples).

All of the dims of `self` must be named in order to use this method. The
resulting tensor is a view on the original tensor.

All dimension names of `self` must be present in `other$names`. `other`
may contain named dimensions that are not in `self$names`; the output
tensor has a size-one dimension for each of those new names.

To align a tensor to a specific order, use `$align_to`.

### Examples:

``` r
# Example 1: Applying a mask
mask <- torch_randint(low = 0, high = 2, size = c(127, 128), dtype=torch_bool())$refine_names(c('W', 'H'))
imgs <- torch_randn(32, 128, 127, 3, names=c('N', 'H', 'W', 'C'))
imgs$masked_fill_(mask$align_as(imgs), 0)

# Example 2: Applying a per-channel-scale
scale_channels <- function(input, scale) {
  scale <- scale$refine_names("C")
  input * scale$align_as(input)
}

num_channels <- 3
scale <- torch_randn(num_channels, names='C')
imgs <- torch_rand(32, 128, 128, num_channels, names=c('N', 'H', 'W', 'C'))
more_imgs = torch_rand(32, num_channels, 128, 128, names=c('N', 'C', 'H', 'W'))
videos = torch_randn(3, num_channels, 128, 128, 128, names=c('N', 'C', 'H', 'W', 'D'))

# scale_channels is agnostic to the dimension order of the input
scale_channels(imgs, scale)
scale_channels(more_imgs, scale)
scale_channels(videos, scale)
```

### Warning:

The named tensor API is experimental and subject to change.

## align_to

Permutes the dimensions of the `self` tensor to match the order
specified in `names`, adding size-one dims for any new names.

All of the dims of `self` must be named in order to use this method. The
resulting tensor is a view on the original tensor.

All dimension names of `self` must be present in `names`. `names` may
contain additional names that are not in `self$names`; the output tensor
has a size-one dimension for each of those new names.

### Arguments:

- names (iterable of str): The desired dimension ordering of the output
  tensor. May contain up to one Ellipsis that is expanded to all
  unmentioned dim names of `self`.

### Examples:

#### Warning:

The named tensor API is experimental and subject to change.

## all

all() -\> bool

Returns TRUE if all elements in the tensor are TRUE, FALSE otherwise.

#### Examples:

``` r
a <- torch_rand(1, 2)$to(dtype = torch_bool())
a
a$all()
```

all(dim, keepdim=FALSE, out=NULL) -\> Tensor

Returns TRUE if all elements in each row of the tensor in the given
dimension `dim` are TRUE, FALSE otherwise.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension `dim` where it is of size 1. Otherwise, `dim` is
squeezed (see `?torch_squeeze()),` resulting in the output tensor having
1 fewer dimension than `input`.

#### Arguments:

- dim (int): the dimension to reduce
- keepdim (bool): whether the output tensor has `dim` retained or not
- out (Tensor, optional): the output tensor

#### Examples:

``` r
a <- torch_rand(4, 2)$to(dtype = torch_bool())
a
a$all(dim=2)
a$all(dim=1)
```

## allclose

allclose(other, rtol=1e-05, atol=1e-08, equal_nan=FALSE) -\> Tensor

See
[`?torch_allclose`](https://torch.mlverse.org/docs/dev/reference/torch_allclose.md)

## angle

angle() -\> Tensor

See
[`?torch_angle`](https://torch.mlverse.org/docs/dev/reference/torch_angle.md)

## any

any() -\> bool

Returns TRUE if any elements in the tensor are TRUE, FALSE otherwise.

#### Examples:

``` r
a <- torch_rand(1, 2)$to(dtype = torch_bool())
a
a$any()
```

any(dim, keepdim=FALSE, out=NULL) -\> Tensor

Returns TRUE if any elements in each row of the tensor in the given
dimension `dim` are TRUE, FALSE otherwise.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension `dim` where it is of size 1. Otherwise, `dim` is
squeezed (see `?torch_squeeze()),` resulting in the output tensor having
1 fewer dimension than `input`.

#### Arguments:

- dim (int): the dimension to reduce
- keepdim (bool): whether the output tensor has `dim` retained or not
- out (Tensor, optional): the output tensor

#### Examples:

``` r
a <- torch_randn(4, 2) < 0
a
a$any(2)
a$any(1)
```

## apply\_

apply\_(callable) -\> Tensor

Applies the function `callable` to each element in the tensor, replacing
each element with the value returned by `callable`.

#### Note:

This function only works with CPU tensors and should not be used in code
sections that require high performance.

## argmax

argmax(dim=NULL, keepdim=FALSE) -\> LongTensor

See
[`?torch_argmax`](https://torch.mlverse.org/docs/dev/reference/torch_argmax.md)

## argmin

argmin(dim=NULL, keepdim=FALSE) -\> LongTensor

See
[`?torch_argmin`](https://torch.mlverse.org/docs/dev/reference/torch_argmin.md)

## argsort

argsort(dim=-1, descending=FALSE) -\> LongTensor

See
[`?torch_argsort`](https://torch.mlverse.org/docs/dev/reference/torch_argsort.md)

## as_strided

as_strided(size, stride, storage_offset=0) -\> Tensor

See \[torch_as_strided()\]

## as_subclass

as_subclass(cls) -\> Tensor

Makes a `cls` instance with the same data pointer as `self`. Changes in
the output mirror changes in `self`, and the output stays attached to
the autograd graph. `cls` must be a subclass of `Tensor`.

## asin

asin() -\> Tensor

See
[`?torch_asin`](https://torch.mlverse.org/docs/dev/reference/torch_asin.md)

## asin\_

asin\_() -\> Tensor

In-place version of `$asin`

## asinh

asinh() -\> Tensor

See
[`?torch_asinh`](https://torch.mlverse.org/docs/dev/reference/torch_asinh.md)

## asinh\_

asinh\_() -\> Tensor

In-place version of `$asinh`

## atan

atan() -\> Tensor

See
[`?torch_atan`](https://torch.mlverse.org/docs/dev/reference/torch_atan.md)

## atan2

atan2(other) -\> Tensor

See \[torch_atan2()\]

## atan2\_

atan2\_(other) -\> Tensor

In-place version of `$atan2`

## atan\_

atan\_() -\> Tensor

In-place version of `$atan`

## atanh

atanh() -\> Tensor

See
[`?torch_atanh`](https://torch.mlverse.org/docs/dev/reference/torch_atanh.md)

## atanh\_

In-place version of `$atanh`

## backward

Computes the gradient of current tensor w.r.t. graph leaves.

The graph is differentiated using the chain rule. If the tensor is
non-scalar (i.e. its data has more than one element) and requires
gradient, the function additionally requires specifying `gradient`. It
should be a tensor of matching type and location, that contains the
gradient of the differentiated function w.r.t. `self`.

This function accumulates gradients in the leaves - you might need to
zero `$grad` attributes or set them to `NULL` before calling it. See
`Default gradient layouts<default-grad-layouts>` for details on the
memory layout of accumulated gradients.

#### Arguments:

- gradient (Tensor or NULL): Gradient w.r.t. the tensor. If it is a
  tensor, it will be automatically converted to a Tensor that does not
  require grad unless `create_graph` is TRUE. NULL values can be
  specified for scalar Tensors or ones that don’t require grad. If a
  NULL value would be acceptable then this argument is optional.
- retain_graph (bool, optional): If `FALSE`, the graph used to compute
  the grads will be freed. Note that in nearly all cases setting this
  option to TRUE is not needed and often can be worked around in a much
  more efficient way. Defaults to the value of `create_graph`.
- create_graph (bool, optional): If `TRUE`, graph of the derivative will
  be constructed, allowing to compute higher order derivative products.
  Defaults to `FALSE`.

## baddbmm

baddbmm(batch1, batch2, \*, beta=1, alpha=1) -\> Tensor

See
[`?torch_baddbmm`](https://torch.mlverse.org/docs/dev/reference/torch_baddbmm.md)

## baddbmm\_

baddbmm\_(batch1, batch2, \*, beta=1, alpha=1) -\> Tensor

In-place version of `$baddbmm`

## bernoulli

bernoulli(\*, generator=NULL) -\> Tensor

Returns a result tensor where each \\\texttt{result\[i\]}\\ is
independently sampled from \\\text{Bernoulli}(\texttt{self\[i\]})\\.
`self` must have floating point `dtype`, and the result will have the
same `dtype`.

See
[`?torch_bernoulli`](https://torch.mlverse.org/docs/dev/reference/torch_bernoulli.md)

## bernoulli\_

bernoulli\_(p=0.5, \*, generator=NULL) -\> Tensor

Fills each location of `self` with an independent sample from
\\\text{Bernoulli}(\texttt{p})\\. `self` can have integral `dtype`.

bernoulli\_(p_tensor, \*, generator=NULL) -\> Tensor

`p_tensor` should be a tensor containing probabilities to be used for
drawing the binary random number.

The \\\text{i}^{th}\\ element of `self` tensor will be set to a value
sampled from \\\text{Bernoulli}(\texttt{p\\tensor\[i\]})\\.

`self` can have integral `dtype`, but `p_tensor` must have floating
point `dtype`.

See also `$bernoulli` and
[`?torch_bernoulli`](https://torch.mlverse.org/docs/dev/reference/torch_bernoulli.md)

## bfloat16

bfloat16(memory_format=torch_preserve_format) -\> Tensor
`self$bfloat16()` is equivalent to `self$to(torch_bfloat16)`. See
\[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## bincount

bincount(weights=NULL, minlength=0) -\> Tensor

See
[`?torch_bincount`](https://torch.mlverse.org/docs/dev/reference/torch_bincount.md)

## bitwise_and

bitwise_and() -\> Tensor

See \[torch_bitwise_and()\]

## bitwise_and\_

bitwise_and\_() -\> Tensor

In-place version of `$bitwise_and`

## bitwise_not

bitwise_not() -\> Tensor

See \[torch_bitwise_not()\]

## bitwise_not\_

bitwise_not\_() -\> Tensor

In-place version of `$bitwise_not`

## bitwise_or

bitwise_or() -\> Tensor

See \[torch_bitwise_or()\]

## bitwise_or\_

bitwise_or\_() -\> Tensor

In-place version of `$bitwise_or`

## bitwise_xor

bitwise_xor() -\> Tensor

See \[torch_bitwise_xor()\]

## bitwise_xor\_

bitwise_xor\_() -\> Tensor

In-place version of `$bitwise_xor`

## bmm

bmm(batch2) -\> Tensor

See
[`?torch_bmm`](https://torch.mlverse.org/docs/dev/reference/torch_bmm.md)

## bool

bool(memory_format=torch_preserve_format) -\> Tensor

`self$bool()` is equivalent to `self$to(torch_bool)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## byte

byte(memory_format=torch_preserve_format) -\> Tensor

`self$byte()` is equivalent to `self$to(torch_uint8)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## cauchy\_

cauchy\_(median=0, sigma=1, \*, generator=NULL) -\> Tensor

Fills the tensor with numbers drawn from the Cauchy distribution:

\\ f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 +
\sigma^2} \\

## ceil

ceil() -\> Tensor

See
[`?torch_ceil`](https://torch.mlverse.org/docs/dev/reference/torch_ceil.md)

## ceil\_

ceil\_() -\> Tensor

In-place version of `$ceil`

## char

char(memory_format=torch_preserve_format) -\> Tensor

`self$char()` is equivalent to `self$to(torch_int8)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## cholesky

cholesky(upper=FALSE) -\> Tensor

See
[`?torch_cholesky`](https://torch.mlverse.org/docs/dev/reference/torch_cholesky.md)

## cholesky_inverse

cholesky_inverse(upper=FALSE) -\> Tensor

See \[torch_cholesky_inverse()\]

## cholesky_solve

cholesky_solve(input2, upper=FALSE) -\> Tensor

See \[torch_cholesky_solve()\]

## chunk

chunk(chunks, dim=0) -\> List of Tensors

See
[`?torch_chunk`](https://torch.mlverse.org/docs/dev/reference/torch_chunk.md)

## clamp

clamp(min, max) -\> Tensor

See
[`?torch_clamp`](https://torch.mlverse.org/docs/dev/reference/torch_clamp.md)

## clamp\_

clamp\_(min, max) -\> Tensor

In-place version of `$clamp`

## clone

clone(memory_format=torch_preserve_format()) -\> Tensor

Returns a copy of the `self` tensor. The copy has the same size and data
type as `self`.

``` r
x <- torch_tensor(1)
y <- x$clone()

x$add_(1)
y
```

#### Note:

Unlike `copy_()`, this function is recorded in the computation graph.
Gradients propagating to the cloned tensor will propagate to the
original tensor.

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of the returned Tensor. Default: `torch_preserve_format`.

## conj

conj() -\> Tensor

See
[`?torch_conj`](https://torch.mlverse.org/docs/dev/reference/torch_conj.md)

## contiguous

contiguous(memory_format=torch_contiguous_format) -\> Tensor

Returns a contiguous in memory tensor containing the same data as `self`
tensor. If `self` tensor is already in the specified memory format, this
function returns the `self` tensor.

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of the returned Tensor. Default: `torch_contiguous_format`.

## copy\_

copy\_(src, non_blocking=FALSE) -\> Tensor

Copies the elements from `src` into `self` tensor and returns `self`.

The `src` tensor must be broadcastable with the `self` tensor. It may be
of a different data type or reside on a different device.

#### Arguments:

- src (Tensor): the source tensor to copy from
- non_blocking (bool): if `TRUE` and this copy is between CPU and GPU,
  the copy may occur asynchronously with respect to the host. For other
  cases, this argument has no effect.

## cos

cos() -\> Tensor

See
[`?torch_cos`](https://torch.mlverse.org/docs/dev/reference/torch_cos.md)

## cos\_

cos\_() -\> Tensor

In-place version of `$cos`

## cosh

cosh() -\> Tensor

See
[`?torch_cosh`](https://torch.mlverse.org/docs/dev/reference/torch_cosh.md)

## cosh\_

cosh\_() -\> Tensor

In-place version of `$cosh`

## cpu

cpu(memory_format=torch_preserve_format) -\> Tensor

Returns a copy of this object in CPU memory.

If this object is already in CPU memory and on the correct device, then
no copy is performed and the original object is returned.

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## cross

cross(other, dim=-1) -\> Tensor

See
[`?torch_cross`](https://torch.mlverse.org/docs/dev/reference/torch_cross.md)

## cuda

cuda(device=NULL, non_blocking=FALSE,
memory_format=torch_preserve_format) -\> Tensor

Returns a copy of this object in CUDA memory.

If this object is already in CUDA memory and on the correct device, then
no copy is performed and the original object is returned.

#### Arguments:

- device (`torch_device`): The destination GPU device. Defaults to the
  current CUDA device.
- non_blocking (bool): If `TRUE` and the source is in pinned memory, the
  copy will be asynchronous with respect to the host. Otherwise, the
  argument has no effect. Default: `FALSE`.
- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## cummax

cummax(dim) -\> (Tensor, Tensor)

See
[`?torch_cummax`](https://torch.mlverse.org/docs/dev/reference/torch_cummax.md)

## cummin

cummin(dim) -\> (Tensor, Tensor)

See
[`?torch_cummin`](https://torch.mlverse.org/docs/dev/reference/torch_cummin.md)

## cumprod

cumprod(dim, dtype=NULL) -\> Tensor

See
[`?torch_cumprod`](https://torch.mlverse.org/docs/dev/reference/torch_cumprod.md)

## cumsum

cumsum(dim, dtype=NULL) -\> Tensor

See
[`?torch_cumsum`](https://torch.mlverse.org/docs/dev/reference/torch_cumsum.md)

## data_ptr

data_ptr() -\> int

Returns the address of the first element of `self` tensor.

## deg2rad

deg2rad() -\> Tensor

See \[torch_deg2rad()\]

## deg2rad\_

deg2rad\_() -\> Tensor

In-place version of `$deg2rad`

## dense_dim

dense_dim() -\> int

If `self` is a sparse COO tensor (i.e., with `torch_sparse_coo` layout),
this returns the number of dense dimensions. Otherwise, this throws an
error.

See also `$sparse_dim`.

## dequantize

dequantize() -\> Tensor

Given a quantized Tensor, dequantize it and return the dequantized float
Tensor.

## det

det() -\> Tensor

See
[`?torch_det`](https://torch.mlverse.org/docs/dev/reference/torch_det.md)

## detach

Returns a new Tensor, detached from the current graph.

The result will never require gradient.

#### Note:

Returned Tensor shares the same storage with the original one.

In-place modifications on either of them will be seen, and may trigger
errors in correctness checks.

**IMPORTANT NOTE**: Previously, in-place size / stride / storage changes
(such as `resize_` / `resize_as_` / `set_` / `transpose_`) to the
returned tensor also update the original tensor. Now, these in-place
changes will not update the original tensor anymore, and will instead
trigger an error.

For sparse tensors: In-place indices / values changes (such as `zero_` /
`copy_` / `add_`) to the returned tensor will not update the original
tensor anymore, and will instead trigger an error.

## detach\_

Detaches the Tensor from the graph that created it, making it a leaf.
Views cannot be detached in-place.

## device

Is the `torch_device` where this Tensor is.

## diag

diag(diagonal=0) -\> Tensor

See
[`?torch_diag`](https://torch.mlverse.org/docs/dev/reference/torch_diag.md)

## diag_embed

diag_embed(offset=0, dim1=-2, dim2=-1) -\> Tensor

See \[torch_diag_embed()\]

## diagflat

diagflat(offset=0) -\> Tensor

See
[`?torch_diagflat`](https://torch.mlverse.org/docs/dev/reference/torch_diagflat.md)

## diagonal

diagonal(offset=0, dim1=0, dim2=1) -\> Tensor

See
[`?torch_diagonal`](https://torch.mlverse.org/docs/dev/reference/torch_diagonal.md)

## digamma

digamma() -\> Tensor

See
[`?torch_digamma`](https://torch.mlverse.org/docs/dev/reference/torch_digamma.md)

## digamma\_

digamma\_() -\> Tensor

In-place version of `$digamma`

## dim

dim() -\> int

Returns the number of dimensions of `self` tensor.

## dist

dist(other, p=2) -\> Tensor

See
[`?torch_dist`](https://torch.mlverse.org/docs/dev/reference/torch_dist.md)

## div

div(value) -\> Tensor

See
[`?torch_div`](https://torch.mlverse.org/docs/dev/reference/torch_div.md)

## div\_

div\_(value) -\> Tensor

In-place version of `$div`

## dot

dot(tensor2) -\> Tensor

See
[`?torch_dot`](https://torch.mlverse.org/docs/dev/reference/torch_dot.md)

## double

double(memory_format=torch_preserve_format) -\> Tensor

`self$double()` is equivalent to `self$to(torch_float64)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## eig

eig(eigenvectors=FALSE) -\> (Tensor, Tensor)

See
[`?torch_eig`](https://torch.mlverse.org/docs/dev/reference/torch_eig.md)

## element_size

element_size() -\> int

Returns the size in bytes of an individual element.

#### Examples:

``` r
torch_tensor(c(1))$element_size()
```

## eq

eq(other) -\> Tensor

See
[`?torch_eq`](https://torch.mlverse.org/docs/dev/reference/torch_eq.md)

## eq\_

eq\_(other) -\> Tensor

In-place version of `$eq`

## equal

equal(other) -\> bool

See
[`?torch_equal`](https://torch.mlverse.org/docs/dev/reference/torch_equal.md)

## erf

erf() -\> Tensor

See
[`?torch_erf`](https://torch.mlverse.org/docs/dev/reference/torch_erf.md)

## erf\_

erf\_() -\> Tensor

In-place version of `$erf`

## erfc

erfc() -\> Tensor

See
[`?torch_erfc`](https://torch.mlverse.org/docs/dev/reference/torch_erfc.md)

## erfc\_

erfc\_() -\> Tensor

In-place version of `$erfc`

## erfinv

erfinv() -\> Tensor

See
[`?torch_erfinv`](https://torch.mlverse.org/docs/dev/reference/torch_erfinv.md)

## erfinv\_

erfinv\_() -\> Tensor

In-place version of `$erfinv`

## exp

exp() -\> Tensor

See
[`?torch_exp`](https://torch.mlverse.org/docs/dev/reference/torch_exp.md)

## exp\_

exp\_() -\> Tensor

In-place version of `$exp`

## expand

expand(\*sizes) -\> Tensor

Returns a new view of the `self` tensor with singleton dimensions
expanded to a larger size.

Passing -1 as the size for a dimension means not changing the size of
that dimension.

Tensor can be also expanded to a larger number of dimensions, and the
new ones will be appended at the front. For the new dimensions, the size
cannot be set to -1.

Expanding a tensor does not allocate new memory, but only creates a new
view on the existing tensor where a dimension of size one is expanded to
a larger size by setting the `stride` to 0. Any dimension of size 1 can
be expanded to an arbitrary value without allocating new memory.

#### Arguments:

- sizes (torch_Size or int…): the desired expanded size

#### Warning:

More than one element of an expanded tensor may refer to a single memory
location. As a result, in-place operations (especially ones that are
vectorized) may result in incorrect behavior. If you need to write to
the tensors, please clone them first.

#### Examples:

``` r
x <- torch_tensor(matrix(c(1,2,3), ncol = 1))
x$size()
x$expand(c(3, 4))
x$expand(c(-1, 4))  # -1 means not changing the size of that dimension
```

## expand_as

expand_as(other) -\> Tensor

Expand this tensor to the same size as `other`. `self$expand_as(other)`
is equivalent to `self$expand(other.size())`.

Please see `$expand` for more information about `expand`.

#### Arguments:

- other (\`\$): The result tensor has the same size
- as `other`.

## expm1

expm1() -\> Tensor

See \[torch_expm1()\]

## expm1\_

expm1\_() -\> Tensor

In-place version of `$expm1`

## exponential\_

exponential\_(lambd=1, \*, generator=NULL) -\> Tensor

Fills `self` tensor with elements drawn from the exponential
distribution:

\\ f(x) = \lambda e^{-\lambda x} \\

## fft

fft(signal_ndim, normalized=FALSE) -\> Tensor

See `?torch_fft`

## fill\_

fill\_(value) -\> Tensor

Fills `self` tensor with the specified value.

## fill_diagonal\_

fill_diagonal\_(fill_value, wrap=FALSE) -\> Tensor

Fill the main diagonal of a tensor that has at least 2-dimensions. When
dims\>2, all dimensions of input must be of equal length. This function
modifies the input tensor in-place, and returns the input tensor.

#### Arguments:

- fill_value (Scalar): the fill value
- wrap (bool): the diagonal ‘wrapped’ after N columns for tall matrices.

#### Examples:

``` r
a <- torch_zeros(3, 3)
a$fill_diagonal_(5)
b <- torch_zeros(7, 3)
b$fill_diagonal_(5)
c <- torch_zeros(7, 3)
c$fill_diagonal_(5, wrap=TRUE)
```

## flatten

flatten(input, start_dim=0, end_dim=-1) -\> Tensor

see
[`?torch_flatten`](https://torch.mlverse.org/docs/dev/reference/torch_flatten.md)

## flip

flip(dims) -\> Tensor

See
[`?torch_flip`](https://torch.mlverse.org/docs/dev/reference/torch_flip.md)

## fliplr

fliplr() -\> Tensor

See
[`?torch_fliplr`](https://torch.mlverse.org/docs/dev/reference/torch_fliplr.md)

## flipud

flipud() -\> Tensor

See
[`?torch_flipud`](https://torch.mlverse.org/docs/dev/reference/torch_flipud.md)

## float

float(memory_format=torch_preserve_format) -\> Tensor

`self$float()` is equivalent to `self$to(torch_float32)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## floor

floor() -\> Tensor

See
[`?torch_floor`](https://torch.mlverse.org/docs/dev/reference/torch_floor.md)

## floor\_

floor\_() -\> Tensor

In-place version of `$floor`

## floor_divide

floor_divide(value) -\> Tensor

See \[torch_floor_divide()\]

## floor_divide\_

floor_divide\_(value) -\> Tensor

In-place version of `$floor_divide`

## fmod

fmod(divisor) -\> Tensor

See
[`?torch_fmod`](https://torch.mlverse.org/docs/dev/reference/torch_fmod.md)

## fmod\_

fmod\_(divisor) -\> Tensor

In-place version of `$fmod`

## frac

frac() -\> Tensor

See
[`?torch_frac`](https://torch.mlverse.org/docs/dev/reference/torch_frac.md)

## frac\_

frac\_() -\> Tensor

In-place version of `$frac`

## gather

gather(dim, index) -\> Tensor

See
[`?torch_gather`](https://torch.mlverse.org/docs/dev/reference/torch_gather.md)

## ge

ge(other) -\> Tensor

See
[`?torch_ge`](https://torch.mlverse.org/docs/dev/reference/torch_ge.md)

## ge\_

ge\_(other) -\> Tensor

In-place version of `$ge`

## geometric\_

geometric\_(p, \*, generator=NULL) -\> Tensor

Fills `self` tensor with elements drawn from the geometric distribution:

\\ f(X=k) = p^{k - 1} (1 - p) \\

## geqrf

geqrf() -\> (Tensor, Tensor)

See
[`?torch_geqrf`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md)

## ger

ger(vec2) -\> Tensor

See
[`?torch_ger`](https://torch.mlverse.org/docs/dev/reference/torch_ger.md)

## get_device

get_device() -\> Device ordinal (Integer)

For CUDA tensors, this function returns the device ordinal of the GPU on
which the tensor resides. For CPU tensors, an error is thrown.

#### Examples:

``` r
x <- torch_randn(3, 4, 5, device='cuda:0')
x$get_device()
x$cpu()$get_device()  # RuntimeError: get_device is not implemented for type torch_FloatTensor
```

## grad

This attribute is `NULL` by default and becomes a Tensor the first time
a call to `backward` computes gradients for `self`. The attribute will
then contain the gradients computed and future calls to \[backward()\]
will accumulate (add) gradients into it.

## gt

gt(other) -\> Tensor

See
[`?torch_gt`](https://torch.mlverse.org/docs/dev/reference/torch_gt.md)

## gt\_

gt\_(other) -\> Tensor

In-place version of `$gt`

## half

half(memory_format=torch_preserve_format) -\> Tensor

`self$half()` is equivalent to `self$to(torch_float16)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## hardshrink

hardshrink(lambd=0.5) -\> Tensor

See \[torch_nn.functional.hardshrink()\]

## has_names

Is `TRUE` if any of this tensor’s dimensions are named. Otherwise, is
`FALSE`.

## histc

histc(bins=100, min=0, max=0) -\> Tensor

See
[`?torch_histc`](https://torch.mlverse.org/docs/dev/reference/torch_histc.md)

## ifft

ifft(signal_ndim, normalized=FALSE) -\> Tensor

See `?torch_ifft`

## imag

Returns a new tensor containing imaginary values of the `self` tensor.
The returned tensor and `self` share the same underlying storage.

#### Warning:

\[imag()\] is only supported for tensors with complex dtypes.

#### Examples:

``` r
x <- torch_randn(4, dtype=torch_cfloat())
x
x$imag
```

## index_add

index_add(tensor1, dim, index, tensor2) -\> Tensor

Out-of-place version of `$index_add_`. `tensor1` corresponds to `self`
in `$index_add_`.

## index_add\_

index_add\_(dim, index, tensor) -\> Tensor

Accumulate the elements of `tensor` into the `self` tensor by adding to
the indices in the order given in `index`. For example, if `dim == 0`
and `index[i] == j`, then the `i` th row of `tensor` is added to the
`j` th row of `self`.

The `dim` th dimension of `tensor` must have the same size as the length
of `index` (which must be a vector), and all other dimensions must match
`self`, or an error will be raised.

#### Note:

In some circumstances when using the CUDA backend with CuDNN, this
operator may select a nondeterministic algorithm to increase
performance. If this is undesirable, you can try to make the operation
deterministic (potentially at a performance cost) by setting
`torch_backends.cudnn.deterministic = TRUE`.

#### Arguments:

- dim (int): dimension along which to index
- index (LongTensor): indices of `tensor` to select from
- tensor (Tensor): the tensor containing values to add

#### Examples:

``` r
x <- torch_ones(5, 3)
t <- torch_tensor(matrix(1:9, ncol = 3), dtype=torch_float())
index <- torch_tensor(c(1L, 4L, 3L))
x$index_add_(1, index, t)
```

## index_copy

index_copy(tensor1, dim, index, tensor2) -\> Tensor

Out-of-place version of `$index_copy_`. `tensor1` corresponds to `self`
in `$index_copy_`.

## index_copy\_

index_copy\_(dim, index, tensor) -\> Tensor

Copies the elements of `tensor` into the `self` tensor by selecting the
indices in the order given in `index`. For example, if `dim == 0` and
`index[i] == j`, then the `i` th row of `tensor` is copied to the `j` th
row of `self`.

The `dim` th dimension of `tensor` must have the same size as the length
of `index` (which must be a vector), and all other dimensions must match
`self`, or an error will be raised.

#### Arguments:

- dim (int): dimension along which to index
- index (LongTensor): indices of `tensor` to select from
- tensor (Tensor): the tensor containing values to copy

#### Examples:

``` r
x <- torch_zeros(5, 3)
t <- torch_tensor(matrix(1:9, ncol = 3), dtype=torch_float())
index <- torch_tensor(c(1, 5, 3))
x$index_copy_(1, index, t)
```

## index_fill

index_fill(tensor1, dim, index, value) -\> Tensor

Out-of-place version of `$index_fill_`. `tensor1` corresponds to `self`
in `$index_fill_`.

## index_fill\_

index_fill\_(dim, index, val) -\> Tensor

Fills the elements of the `self` tensor with value `val` by selecting
the indices in the order given in `index`.

#### Arguments:

- dim (int): dimension along which to index
- index (LongTensor): indices of `self` tensor to fill in
- val (float): the value to fill with

#### Examples:

``` r
x <- torch_tensor(matrix(1:9, ncol = 3), dtype=torch_float())
index <- torch_tensor(c(1, 3), dtype = torch_long())
x$index_fill_(1, index, -1)
```

## index_put

index_put(tensor1, indices, value, accumulate=FALSE) -\> Tensor

Out-place version of `$index_put_`. `tensor1` corresponds to `self` in
`$index_put_`.

## index_put\_

index_put\_(indices, value, accumulate=FALSE) -\> Tensor

Puts values from the tensor `value` into the tensor `self` using the
indices specified in `indices` (which is a tuple of Tensors). The
expression `tensor.index_put_(indices, value)` is equivalent to
`tensor[indices] = value`. Returns `self`.

If `accumulate` is `TRUE`, the elements in `value` are added to `self`.
If accumulate is `FALSE`, the behavior is undefined if indices contain
duplicate elements.

#### Arguments:

- indices (tuple of LongTensor): tensors used to index into `self`.
- value (Tensor): tensor of same dtype as `self`.
- accumulate (bool): whether to accumulate into self

## index_select

index_select(dim, index) -\> Tensor

See \[torch_index_select()\]

## indices

indices() -\> Tensor

If `self` is a sparse COO tensor (i.e., with `torch_sparse_coo` layout),
this returns a view of the contained indices tensor. Otherwise, this
throws an error.

See also `Tensor.values`.

#### Note:

This method can only be called on a coalesced sparse tensor. See
`Tensor.coalesce` for details.

## int

int(memory_format=torch_preserve_format) -\> Tensor

`self$int()` is equivalent to `self$to(torch_int32)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## int_repr

int_repr() -\> Tensor

Given a quantized Tensor, `self$int_repr()` returns a CPU Tensor with
uint8_t as data type that stores the underlying uint8_t values of the
given Tensor.

## inverse

inverse() -\> Tensor

See
[`?torch_inverse`](https://torch.mlverse.org/docs/dev/reference/torch_inverse.md)

## irfft

irfft(signal_ndim, normalized=FALSE, onesided=TRUE, signal_sizes=NULL)
-\> Tensor

See `?torch_irfft`

## is_complex

is_complex() -\> bool

Returns TRUE if the data type of `self` is a complex data type.

## is_contiguous

is_contiguous(memory_format=torch_contiguous_format) -\> bool

Returns TRUE if `self` tensor is contiguous in memory in the order
specified by memory format.

#### Arguments:

- memory_format (`torch_memory_format`, optional): Specifies memory
  allocation
- order. Default: `torch_contiguous_format`.

## is_cuda

Is `TRUE` if the Tensor is stored on the GPU, `FALSE` otherwise.

## is_floating_point

is_floating_point() -\> bool

Returns TRUE if the data type of `self` is a floating point data type.

## is_leaf

All Tensors that have `requires_grad` which is `FALSE` will be leaf
Tensors by convention.

For Tensors that have `requires_grad` which is `TRUE`, they will be leaf
Tensors if they were created by the user. This means that they are not
the result of an operation and so `grad_fn` is NULL.

Only leaf Tensors will have their `grad` populated during a call to
\[backward()\]. To get `grad` populated for non-leaf Tensors, you can
use \[retain_grad()\].

#### Examples:

``` r
a <- torch_rand(10, requires_grad=TRUE)
a$is_leaf

# b <- torch_rand(10, requires_grad=TRUE)$cuda()
# b$is_leaf()
# FALSE
# b was created by the operation that cast a cpu Tensor into a cuda Tensor

c <- torch_rand(10, requires_grad=TRUE) + 2
c$is_leaf
# c was created by the addition operation

# d <- torch_rand(10)$cuda()
# d$is_leaf()
# TRUE
# d does not require gradients and so has no operation creating it (that is tracked by the autograd engine)

# e <- torch_rand(10)$cuda()$requires_grad_()
# e$is_leaf()
# TRUE
# e requires gradients and has no operations creating it

# f <- torch_rand(10, requires_grad=TRUE, device="cuda")
# f$is_leaf
# TRUE
# f requires grad, has no operation creating it
```

## is_meta

Is `TRUE` if the Tensor is a meta tensor, `FALSE` otherwise. Meta
tensors are like normal tensors, but they carry no data.

## is_pinned

Returns true if this tensor resides in pinned memory.

## is_quantized

Is `TRUE` if the Tensor is quantized, `FALSE` otherwise.

## is_set_to

is_set_to(tensor) -\> bool

Returns TRUE if this object refers to the same `THTensor` object from
the Torch C API as the given tensor.

## is_shared

Checks if tensor is in shared memory.

This is always `TRUE` for CUDA tensors.

## is_signed

is_signed() -\> bool

Returns TRUE if the data type of `self` is a signed data type.

## isclose

isclose(other, rtol=1e-05, atol=1e-08, equal_nan=FALSE) -\> Tensor

See
[`?torch_isclose`](https://torch.mlverse.org/docs/dev/reference/torch_isclose.md)

## isfinite

isfinite() -\> Tensor

See
[`?torch_isfinite`](https://torch.mlverse.org/docs/dev/reference/torch_isfinite.md)

## isinf

isinf() -\> Tensor

See
[`?torch_isinf`](https://torch.mlverse.org/docs/dev/reference/torch_isinf.md)

## isnan

isnan() -\> Tensor

See
[`?torch_isnan`](https://torch.mlverse.org/docs/dev/reference/torch_isnan.md)

## istft

See
[`?torch_istft`](https://torch.mlverse.org/docs/dev/reference/torch_istft.md)
\## item

item() -\> number

Returns the value of this tensor as a standard Python number. This only
works for tensors with one element. For other cases, see `$tolist`.

This operation is not differentiable.

#### Examples:

``` r
x <- torch_tensor(1.0)
x$item()
```

## kthvalue

kthvalue(k, dim=NULL, keepdim=FALSE) -\> (Tensor, LongTensor)

See
[`?torch_kthvalue`](https://torch.mlverse.org/docs/dev/reference/torch_kthvalue.md)

## le

le(other) -\> Tensor

See
[`?torch_le`](https://torch.mlverse.org/docs/dev/reference/torch_le.md)

## le\_

le\_(other) -\> Tensor

In-place version of `$le`

## lerp

lerp(end, weight) -\> Tensor

See
[`?torch_lerp`](https://torch.mlverse.org/docs/dev/reference/torch_lerp.md)

## lerp\_

lerp\_(end, weight) -\> Tensor

In-place version of `$lerp`

## lgamma

lgamma() -\> Tensor

See
[`?torch_lgamma`](https://torch.mlverse.org/docs/dev/reference/torch_lgamma.md)

## lgamma\_

lgamma\_() -\> Tensor

In-place version of `$lgamma`

## log

log() -\> Tensor

See
[`?torch_log`](https://torch.mlverse.org/docs/dev/reference/torch_log.md)

## log10

log10() -\> Tensor

See \[torch_log10()\]

## log10\_

log10\_() -\> Tensor

In-place version of `$log10`

## log1p

log1p() -\> Tensor

See \[torch_log1p()\]

## log1p\_

log1p\_() -\> Tensor

In-place version of `$log1p`

## log2

log2() -\> Tensor

See \[torch_log2()\]

## log2\_

log2\_() -\> Tensor

In-place version of `$log2`

## log\_

log\_() -\> Tensor

In-place version of `$log`

## log_normal\_

log_normal\_(mean=1, std=2, \*, generator=NULL)

Fills `self` tensor with numbers samples from the log-normal
distribution parameterized by the given mean `\mu` and standard
deviation `\sigma`. Note that `mean` and `std` are the mean and standard
deviation of the underlying normal distribution, and not of the returned
distribution:

\\ f(x) = \dfrac{1}{x \sigma \sqrt{2\pi}}\\ e^{-\frac{(\ln x -
\mu)^2}{2\sigma^2}} \\

## logaddexp

logaddexp(other) -\> Tensor

See
[`?torch_logaddexp`](https://torch.mlverse.org/docs/dev/reference/torch_logaddexp.md)

## logaddexp2

logaddexp2(other) -\> Tensor

See \[torch_logaddexp2()\]

## logcumsumexp

logcumsumexp(dim) -\> Tensor

See
[`?torch_logcumsumexp`](https://torch.mlverse.org/docs/dev/reference/torch_logcumsumexp.md)

## logdet

logdet() -\> Tensor

See
[`?torch_logdet`](https://torch.mlverse.org/docs/dev/reference/torch_logdet.md)

## logical_and

logical_and() -\> Tensor

See \[torch_logical_and()\]

## logical_and\_

logical_and\_() -\> Tensor

In-place version of `$logical_and`

## logical_not

logical_not() -\> Tensor

See \[torch_logical_not()\]

## logical_not\_

logical_not\_() -\> Tensor

In-place version of `$logical_not`

## logical_or

logical_or() -\> Tensor

See \[torch_logical_or()\]

## logical_or\_

logical_or\_() -\> Tensor

In-place version of `$logical_or`

## logical_xor

logical_xor() -\> Tensor

See \[torch_logical_xor()\]

## logical_xor\_

logical_xor\_() -\> Tensor

In-place version of `$logical_xor`

## logsumexp

logsumexp(dim, keepdim=FALSE) -\> Tensor

See
[`?torch_logsumexp`](https://torch.mlverse.org/docs/dev/reference/torch_logsumexp.md)

## long

long(memory_format=torch_preserve_format) -\> Tensor

`self$long()` is equivalent to `self$to(torch_int64)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## lstsq

lstsq(A) -\> (Tensor, Tensor)

See
[`?torch_lstsq`](https://torch.mlverse.org/docs/dev/reference/torch_lstsq.md)

## lt

lt(other) -\> Tensor

See
[`?torch_lt`](https://torch.mlverse.org/docs/dev/reference/torch_lt.md)

## lt\_

lt\_(other) -\> Tensor

In-place version of `$lt`

## lu

See
[`?torch_lu`](https://torch.mlverse.org/docs/dev/reference/torch_lu.md)
\## lu_solve

lu_solve(LU_data, LU_pivots) -\> Tensor

See \[torch_lu_solve()\]

## map\_

map\_(tensor, callable)

Applies `callable` for each element in `self` tensor and the given
`tensor` and stores the results in `self` tensor. `self` tensor and the
given `tensor` must be broadcastable.

The `callable` should have the signature:

`callable(a, b) -> number`

## masked_fill

masked_fill(mask, value) -\> Tensor

Out-of-place version of `$masked_fill_`

## masked_fill\_

masked_fill\_(mask, value)

Fills elements of `self` tensor with `value` where `mask` is TRUE. The
shape of `mask` must be `broadcastable <broadcasting-semantics>` with
the shape of the underlying tensor.

#### Arguments:

- mask (BoolTensor): the boolean mask
- value (float): the value to fill in with

## masked_scatter

masked_scatter(mask, tensor) -\> Tensor

Out-of-place version of `$masked_scatter_`

## masked_scatter\_

masked_scatter\_(mask, source)

Copies elements from `source` into `self` tensor at positions where the
`mask` is TRUE. The shape of `mask` must be
:ref:`broadcastable <broadcasting-semantics>` with the shape of the
underlying tensor. The `source` should have at least as many elements as
the number of ones in `mask`

#### Arguments:

- mask (BoolTensor): the boolean mask
- source (Tensor): the tensor to copy from

#### Note:

The `mask` operates on the `self` tensor, not on the given `source`
tensor.

## masked_select

masked_select(mask) -\> Tensor

See \[torch_masked_select()\]

## matmul

matmul(tensor2) -\> Tensor

See
[`?torch_matmul`](https://torch.mlverse.org/docs/dev/reference/torch_matmul.md)

## matrix_power

matrix_power(n) -\> Tensor

See \[torch_matrix_power()\]

## max

max(dim=NULL, keepdim=FALSE) -\> Tensor or (Tensor, Tensor)

See
[`?torch_max`](https://torch.mlverse.org/docs/dev/reference/torch_max.md)

## mean

mean(dim=NULL, keepdim=FALSE) -\> Tensor or (Tensor, Tensor)

See
[`?torch_mean`](https://torch.mlverse.org/docs/dev/reference/torch_mean.md)

## median

median(dim=NULL, keepdim=FALSE) -\> (Tensor, LongTensor)

See
[`?torch_median`](https://torch.mlverse.org/docs/dev/reference/torch_median.md)

## min

min(dim=NULL, keepdim=FALSE) -\> Tensor or (Tensor, Tensor)

See
[`?torch_min`](https://torch.mlverse.org/docs/dev/reference/torch_min.md)

## mm

mm(mat2) -\> Tensor

See
[`?torch_mm`](https://torch.mlverse.org/docs/dev/reference/torch_mm.md)

## mode

mode(dim=NULL, keepdim=FALSE) -\> (Tensor, LongTensor)

See
[`?torch_mode`](https://torch.mlverse.org/docs/dev/reference/torch_mode.md)

## mul

mul(value) -\> Tensor

See
[`?torch_mul`](https://torch.mlverse.org/docs/dev/reference/torch_mul.md)

## mul\_

mul\_(value)

In-place version of `$mul`

## multinomial

multinomial(num_samples, replacement=FALSE, \*, generator=NULL) -\>
Tensor

See
[`?torch_multinomial`](https://torch.mlverse.org/docs/dev/reference/torch_multinomial.md)

## mv

mv(vec) -\> Tensor

See
[`?torch_mv`](https://torch.mlverse.org/docs/dev/reference/torch_mv.md)

## mvlgamma

mvlgamma(p) -\> Tensor

See
[`?torch_mvlgamma`](https://torch.mlverse.org/docs/dev/reference/torch_mvlgamma.md)

## mvlgamma\_

mvlgamma\_(p) -\> Tensor

In-place version of `$mvlgamma`

## names

Stores names for each of this tensor’s dimensions.

`names[idx]` corresponds to the name of tensor dimension `idx`. Names
are either a string if the dimension is named or `NULL` if the dimension
is unnamed.

Dimension names may contain characters or underscore. Furthermore, a
dimension name must be a valid Python variable name (i.e., does not
start with underscore).

Tensors may not have two named dimensions with the same name.

#### Warning:

The named tensor API is experimental and subject to change.

## narrow

narrow(dimension, start, length) -\> Tensor

See
[`?torch_narrow`](https://torch.mlverse.org/docs/dev/reference/torch_narrow.md)

#### Examples:

``` r
x <- torch_tensor(matrix(1:9, ncol = 3))
x$narrow(1, 1, 3)
x$narrow(1, 1, 2)
```

## narrow_copy

narrow_copy(dimension, start, length) -\> Tensor

Same as `Tensor.narrow` except returning a copy rather than shared
storage. This is primarily for sparse tensors, which do not have a
shared-storage narrow method. Calling `` narrow_copy` with ``dimemsion
\> self\\sparse_dim()\`\` will return a copy with the relevant dense
dimension narrowed, and \`\`self\\shape\`\` updated accordingly.

## ndim

Alias for `$dim()`

## ndimension

ndimension() -\> int

Alias for `$dim()`

## ne

ne(other) -\> Tensor

See
[`?torch_ne`](https://torch.mlverse.org/docs/dev/reference/torch_ne.md)

## ne\_

ne\_(other) -\> Tensor

In-place version of `$ne`

## neg

neg() -\> Tensor

See
[`?torch_neg`](https://torch.mlverse.org/docs/dev/reference/torch_neg.md)

## neg\_

neg\_() -\> Tensor

In-place version of `$neg`

## nelement

nelement() -\> int

Alias for `$numel`

## new_empty

new_empty(size, dtype=NULL, device=NULL, requires_grad=FALSE) -\> Tensor

Returns a Tensor of size `size` filled with uninitialized data. By
default, the returned Tensor has the same `torch_dtype` and
`torch_device` as this tensor.

#### Arguments:

- dtype (`torch_dtype`, optional): the desired type of returned tensor.
  Default: if NULL, same `torch_dtype` as this tensor.
- device (`torch_device`, optional): the desired device of returned
  tensor. Default: if NULL, same `torch_device` as this tensor.
- requires_grad (bool, optional): If autograd should record operations
  on the
- returned tensor. Default: `FALSE`.

#### Examples:

``` r
tensor <- torch_ones(5)
tensor$new_empty(c(2, 3))
```

## new_full

new_full(size, fill_value, dtype=NULL, device=NULL, requires_grad=FALSE)
-\> Tensor

Returns a Tensor of size `size` filled with `fill_value`. By default,
the returned Tensor has the same `torch_dtype` and `torch_device` as
this tensor.

#### Arguments:

- fill_value (scalar): the number to fill the output tensor with.
- dtype (`torch_dtype`, optional): the desired type of returned tensor.
  Default: if NULL, same `torch_dtype` as this tensor.
- device (`torch_device`, optional): the desired device of returned
  tensor. Default: if NULL, same `torch_device` as this tensor.
- requires_grad (bool, optional): If autograd should record operations
  on the
- returned tensor. Default: `FALSE`.

#### Examples:

``` r
tensor <- torch_ones(c(2), dtype=torch_float64())
tensor$new_full(c(3, 4), 3.141592)
```

## new_ones

new_ones(size, dtype=NULL, device=NULL, requires_grad=FALSE) -\> Tensor

Returns a Tensor of size `size` filled with `1`. By default, the
returned Tensor has the same `torch_dtype` and `torch_device` as this
tensor.

#### Arguments:

- size (int…): a list, tuple, or `torch_Size` of integers defining the
- shape of the output tensor.
- dtype (`torch_dtype`, optional): the desired type of returned tensor.
  Default: if NULL, same `torch_dtype` as this tensor.
- device (`torch_device`, optional): the desired device of returned
  tensor. Default: if NULL, same `torch_device` as this tensor.
- requires_grad (bool, optional): If autograd should record operations
  on the
- returned tensor. Default: `FALSE`.

#### Examples:

``` r
tensor <- torch_tensor(c(2), dtype=torch_int32())
tensor$new_ones(c(2, 3))
```

## new_tensor

new_tensor(data, dtype=NULL, device=NULL, requires_grad=FALSE) -\>
Tensor

Returns a new Tensor with `data` as the tensor data. By default, the
returned Tensor has the same `torch_dtype` and `torch_device` as this
tensor.

#### Warning:

`new_tensor` always copies `data(). If you have a Tensor`data\` and want
to avoid a copy, use \[\\requires_grad\_()\] or \[\\detach()\]. If you
have a numpy array and want to avoid a copy, use \[torch_from_numpy()\].

When data is a tensor `x`, \[new_tensor()()\] reads out ‘the data’ from
whatever it is passed, and constructs a leaf variable. Therefore
`tensor$new_tensor(x)` is equivalent to `x$clone()$detach()` and
`tensor$new_tensor(x, requires_grad=TRUE)` is equivalent to
`x$clone()$detach()$requires_grad_(TRUE)`. The equivalents using
`clone()` and [`detach()`](https://rdrr.io/r/base/detach.html) are
recommended.

#### Arguments:

- data (array_like): The returned Tensor copies `data`.
- dtype (`torch_dtype`, optional): the desired type of returned tensor.
  Default: if NULL, same `torch_dtype` as this tensor.
- device (`torch_device`, optional): the desired device of returned
  tensor. Default: if NULL, same `torch_device` as this tensor.
- requires_grad (bool, optional): If autograd should record operations
  on the
- returned tensor. Default: `FALSE`.

#### Examples:

``` r
tensor <- torch_ones(c(2), dtype=torch_int8)
data <- matrix(1:4, ncol = 2)
tensor$new_tensor(data)
```

## new_zeros

new_zeros(size, dtype=NULL, device=NULL, requires_grad=FALSE) -\> Tensor

Returns a Tensor of size `size` filled with `0`. By default, the
returned Tensor has the same `torch_dtype` and `torch_device` as this
tensor.

#### Arguments:

- size (int…): a list, tuple, or `torch_Size` of integers defining the
- shape of the output tensor.
- dtype (`torch_dtype`, optional): the desired type of returned tensor.
  Default: if NULL, same `torch_dtype` as this tensor.
- device (`torch_device`, optional): the desired device of returned
  tensor. Default: if NULL, same `torch_device` as this tensor.
- requires_grad (bool, optional): If autograd should record operations
  on the
- returned tensor. Default: `FALSE`.

#### Examples:

``` r
tensor <- torch_tensor(c(1), dtype=torch_float64())
tensor$new_zeros(c(2, 3))
```

## nonzero

nonzero() -\> LongTensor

See
[`?torch_nonzero`](https://torch.mlverse.org/docs/dev/reference/torch_nonzero.md)

## norm

See
[`?torch_norm`](https://torch.mlverse.org/docs/dev/reference/torch_norm.md)
\## normal\_

normal\_(mean=0, std=1, \*, generator=NULL) -\> Tensor

Fills `self` tensor with elements samples from the normal distribution
parameterized by `mean` and `std`.

## numel

numel() -\> int

See `?torch_numel`

## numpy

numpy() -\> numpy.ndarray

Returns `self` tensor as a NumPy :class:`ndarray`. This tensor and the
returned `ndarray` share the same underlying storage. Changes to `self`
tensor will be reflected in the :class:`ndarray` and vice versa.

## orgqr

orgqr(input2) -\> Tensor

See
[`?torch_orgqr`](https://torch.mlverse.org/docs/dev/reference/torch_orgqr.md)

## ormqr

ormqr(input2, input3, left=TRUE, transpose=FALSE) -\> Tensor

See
[`?torch_ormqr`](https://torch.mlverse.org/docs/dev/reference/torch_ormqr.md)

## permute

permute(\*dims) -\> Tensor

Returns a view of the original tensor with its dimensions permuted.

#### Arguments:

- dims (int…): The desired ordering of dimensions

#### Examples:

``` r
x <- torch_randn(2, 3, 5)
x$size()
x$permute(c(3, 1, 2))$size()
```

## pin_memory

pin_memory() -\> Tensor

Copies the tensor to pinned memory, if it’s not already pinned.

## pinverse

pinverse() -\> Tensor

See
[`?torch_pinverse`](https://torch.mlverse.org/docs/dev/reference/torch_pinverse.md)

## polygamma

polygamma(n) -\> Tensor

See
[`?torch_polygamma`](https://torch.mlverse.org/docs/dev/reference/torch_polygamma.md)

## polygamma\_

polygamma\_(n) -\> Tensor

In-place version of `$polygamma`

## pow

pow(exponent) -\> Tensor

See
[`?torch_pow`](https://torch.mlverse.org/docs/dev/reference/torch_pow.md)

## pow\_

pow\_(exponent) -\> Tensor

In-place version of `$pow`

## prod

prod(dim=NULL, keepdim=FALSE, dtype=NULL) -\> Tensor

See
[`?torch_prod`](https://torch.mlverse.org/docs/dev/reference/torch_prod.md)

## put\_

put\_(indices, tensor, accumulate=FALSE) -\> Tensor

Copies the elements from `tensor` into the positions specified by
indices. For the purpose of indexing, the `self` tensor is treated as if
it were a 1-D tensor.

If `accumulate` is `TRUE`, the elements in `tensor` are added to `self`.
If accumulate is `FALSE`, the behavior is undefined if indices contain
duplicate elements.

#### Arguments:

- indices (LongTensor): the indices into self
- tensor (Tensor): the tensor containing values to copy from
- accumulate (bool): whether to accumulate into self

#### Examples:

``` r
src <- torch_tensor(matrix(3:8, ncol = 3))
src$put_(torch_tensor(1:2), torch_tensor(9:10))
```

## q_per_channel_axis

q_per_channel_axis() -\> int

Given a Tensor quantized by linear (affine) per-channel quantization,
returns the index of dimension on which per-channel quantization is
applied.

## q_per_channel_scales

q_per_channel_scales() -\> Tensor

Given a Tensor quantized by linear (affine) per-channel quantization,
returns a Tensor of scales of the underlying quantizer. It has the
number of elements that matches the corresponding dimensions (from
q_per_channel_axis) of the tensor.

## q_per_channel_zero_points

q_per_channel_zero_points() -\> Tensor

Given a Tensor quantized by linear (affine) per-channel quantization,
returns a tensor of zero_points of the underlying quantizer. It has the
number of elements that matches the corresponding dimensions (from
q_per_channel_axis) of the tensor.

## q_scale

q_scale() -\> float

Given a Tensor quantized by linear(affine) quantization, returns the
scale of the underlying quantizer().

## q_zero_point

q_zero_point() -\> int

Given a Tensor quantized by linear(affine) quantization, returns the
zero_point of the underlying quantizer().

## qr

qr(some=TRUE) -\> (Tensor, Tensor)

See
[`?torch_qr`](https://torch.mlverse.org/docs/dev/reference/torch_qr.md)

## qscheme

qscheme() -\> torch_qscheme

Returns the quantization scheme of a given QTensor.

## rad2deg

rad2deg() -\> Tensor

See \[torch_rad2deg()\]

## rad2deg\_

rad2deg\_() -\> Tensor

In-place version of `$rad2deg`

## random\_

random\_(from=0, to=NULL, \*, generator=NULL) -\> Tensor

Fills `self` tensor with numbers sampled from the discrete uniform
distribution over `[from, to - 1]`. If not specified, the values are
usually only bounded by `self` tensor’s data type. However, for floating
point types, if unspecified, range will be `[0, 2^mantissa]` to ensure
that every value is representable. For example,
`torch_tensor(1, dtype=torch_double).random_()` will be uniform in
`[0, 2^53]`.

## real

Returns a new tensor containing real values of the `self` tensor. The
returned tensor and `self` share the same underlying storage.

#### Warning:

\[real()\] is only supported for tensors with complex dtypes.

#### Examples:

``` r
x <- torch_randn(4, dtype=torch_cfloat())
x
x$real
```

## reciprocal

reciprocal() -\> Tensor

See
[`?torch_reciprocal`](https://torch.mlverse.org/docs/dev/reference/torch_reciprocal.md)

## reciprocal\_

reciprocal\_() -\> Tensor

In-place version of `$reciprocal`

## record_stream

record_stream(stream)

Ensures that the tensor memory is not reused for another tensor until
all current work queued on `stream` are complete.

#### Note:

The caching allocator is aware of only the stream where a tensor was
allocated. Due to the awareness, it already correctly manages the life
cycle of tensors on only one stream. But if a tensor is used on a stream
different from the stream of origin, the allocator might reuse the
memory unexpectedly. Calling this method lets the allocator know which
streams have used the tensor.

## refine_names

Refines the dimension names of `self` according to `names`.

Refining is a special case of renaming that “lifts” unnamed dimensions.
A `NULL` dim can be refined to have any name; a named dim can only be
refined to have the same name.

Because named tensors can coexist with unnamed tensors, refining names
gives a nice way to write named-tensor-aware code that works with both
named and unnamed tensors.

`names` may contain up to one Ellipsis (`...`). The Ellipsis is expanded
greedily; it is expanded in-place to fill `names` to the same length as
`self$dim()` using names from the corresponding indices of `self$names`.

#### Arguments:

- names (iterable of str): The desired names of the output tensor. May
  contain up to one Ellipsis.

#### Examples:

``` r
imgs <- torch_randn(32, 3, 128, 128)
named_imgs <- imgs$refine_names(c('N', 'C', 'H', 'W'))
named_imgs$names
```

## register_hook

Registers a backward hook.

The hook will be called every time a gradient with respect to the Tensor
is computed. The hook should have the following signature::

hook(grad) -\> Tensor or NULL

The hook should not modify its argument, but it can optionally return a
new gradient which will be used in place of `grad`.

This function returns a handle with a method `handle$remove()` that
removes the hook from the module.

#### Example

``` r
v <- torch_tensor(c(0., 0., 0.), requires_grad=TRUE)
h <- v$register_hook(function(grad) grad * 2)  # double the gradient
v$backward(torch_tensor(c(1., 2., 3.)))
v$grad
h$remove()
```

## remainder

remainder(divisor) -\> Tensor

See
[`?torch_remainder`](https://torch.mlverse.org/docs/dev/reference/torch_remainder.md)

## remainder\_

remainder\_(divisor) -\> Tensor

In-place version of `$remainder`

## rename

Renames dimension names of `self`.

There are two main usages:

`self$rename(**rename_map)` returns a view on tensor that has dims
renamed as specified in the mapping `rename_map`.

`self$rename(*names)` returns a view on tensor, renaming all dimensions
positionally using `names`. Use `self$rename(NULL)` to drop names on a
tensor.

One cannot specify both positional args `names` and keyword args
`rename_map`.

#### Examples:

``` r
imgs <- torch_rand(2, 3, 5, 7, names=c('N', 'C', 'H', 'W'))
renamed_imgs <- imgs$rename(c("Batch", "Channels", "Height", "Width"))
```

## rename\_

In-place version of `$rename`.

## renorm

renorm(p, dim, maxnorm) -\> Tensor

See
[`?torch_renorm`](https://torch.mlverse.org/docs/dev/reference/torch_renorm.md)

## renorm\_

renorm\_(p, dim, maxnorm) -\> Tensor

In-place version of `$renorm`

## repeat

repeat(\*sizes) -\> Tensor

Repeats this tensor along the specified dimensions.

Unlike `$expand`, this function copies the tensor’s data.

#### Arguments:

- sizes (torch_Size or int…): The number of times to repeat this tensor
  along each
- dimension

#### Examples:

``` r
x <- torch_tensor(c(1, 2, 3))
x$`repeat`(c(4, 2))
x$`repeat`(c(4, 2, 1))$size()
```

## repeat_interleave

repeat_interleave(repeats, dim=NULL) -\> Tensor

See \[torch_repeat_interleave()\].

## requires_grad

Is `TRUE` if gradients need to be computed for this Tensor, `FALSE`
otherwise.

#### Note:

The fact that gradients need to be computed for a Tensor do not mean
that the `grad` attribute will be populated, see `is_leaf` for more
details.

## requires_grad\_

requires_grad\_(requires_grad=TRUE) -\> Tensor

Change if autograd should record operations on this tensor: sets this
tensor’s `requires_grad` attribute in-place. Returns this tensor.

\[requires_grad\_()\]’s main use case is to tell autograd to begin
recording operations on a Tensor `tensor`. If `tensor` has
`requires_grad=FALSE` (because it was obtained through a DataLoader, or
required preprocessing or initialization), `tensor.requires_grad_()`
makes it so that autograd will begin to record operations on `tensor`.

#### Arguments:

- requires_grad (bool): If autograd should record operations on this
  tensor. Default: `TRUE`.

#### Examples:

``` r
# Let's say we want to preprocess some saved weights and use
# the result as new weights.
saved_weights <- c(0.1, 0.2, 0.3, 0.25)
loaded_weights <- torch_tensor(saved_weights)
weights <- preprocess(loaded_weights)  # some function
weights

# Now, start to record operations done to weights
weights$requires_grad_()
out <- weights$pow(2)$sum()
out$backward()
weights$grad
```

## reshape

reshape(\*shape) -\> Tensor

Returns a tensor with the same data and number of elements as `self` but
with the specified shape. This method returns a view if `shape` is
compatible with the current shape. See `$view` on when it is possible to
return a view.

See
[`?torch_reshape`](https://torch.mlverse.org/docs/dev/reference/torch_reshape.md)

#### Arguments:

- shape (tuple of ints or int…): the desired shape

## reshape_as

reshape_as(other) -\> Tensor

Returns this tensor as the same shape as `other`.
`self$reshape_as(other)` is equivalent to `self$reshape(other.sizes())`.
This method returns a view if `other.sizes()` is compatible with the
current shape. See `$view` on when it is possible to return a view.

Please see `reshape` for more information about `reshape`.

#### Arguments:

- other (\`\$): The result tensor has the same shape
- as `other`.

## resize\_

resize\_(\*sizes, memory_format=torch_contiguous_format) -\> Tensor

Resizes `self` tensor to the specified size. If the number of elements
is larger than the current storage size, then the underlying storage is
resized to fit the new number of elements. If the number of elements is
smaller, the underlying storage is not changed. Existing elements are
preserved but any new memory is uninitialized.

#### Warning:

This is a low-level method. The storage is reinterpreted as
C-contiguous, ignoring the current strides (unless the target size
equals the current size, in which case the tensor is left unchanged).
For most purposes, you will instead want to use `$view()`, which checks
for contiguity, or `$reshape()`, which copies data if needed. To change
the size in-place with custom strides, see `$set_()`.

#### Arguments:

- sizes (torch_Size or int…): the desired size
- memory_format (`torch_memory_format`, optional): the desired memory
  format of Tensor. Default: `torch_contiguous_format`. Note that memory
  format of `self` is going to be unaffected if `self$size()` matches
  `sizes`.

#### Examples:

``` r
x <- torch_tensor(matrix(1:6, ncol = 2))
x$resize_(c(2, 2))
```

## resize_as\_

resize_as\_(tensor, memory_format=torch_contiguous_format) -\> Tensor

Resizes the `self` tensor to be the same size as the specified `tensor`.
This is equivalent to `self$resize_(tensor.size())`.

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of Tensor. Default: `torch_contiguous_format`. Note that memory
  format of `self` is going to be unaffected if `self$size()` matches
  `tensor.size()`.

## retain_grad

Enables `$grad` attribute for non-leaf Tensors.

## rfft

rfft(signal_ndim, normalized=FALSE, onesided=TRUE) -\> Tensor

See `?torch_rfft`

## roll

roll(shifts, dims) -\> Tensor

See
[`?torch_roll`](https://torch.mlverse.org/docs/dev/reference/torch_roll.md)

## rot90

rot90(k, dims) -\> Tensor

See \[torch_rot90()\]

## round

round() -\> Tensor

See
[`?torch_round`](https://torch.mlverse.org/docs/dev/reference/torch_round.md)

## round\_

round\_() -\> Tensor

In-place version of `$round`

## rsqrt

rsqrt() -\> Tensor

See
[`?torch_rsqrt`](https://torch.mlverse.org/docs/dev/reference/torch_rsqrt.md)

## rsqrt\_

rsqrt\_() -\> Tensor

In-place version of `$rsqrt`

## scatter

scatter(dim, index, src) -\> Tensor

Out-of-place version of `$scatter_`

## scatter\_

scatter\_(dim, index, src) -\> Tensor

Writes all values from the tensor `src` into `self` at the indices
specified in the `index` tensor. For each value in `src`, its output
index is specified by its index in `src` for `dimension != dim` and by
the corresponding value in `index` for `dimension = dim`.

For a 3-D tensor, `self` is updated as:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

This is the reverse operation of the manner described in `$gather`.

`self`, `index` and `src` (if it is a Tensor) should have same number of
dimensions. It is also required that `index.size(d) <= src.size(d)` for
all dimensions `d`, and that `index.size(d) <= self$size(d)` for all
dimensions `d != dim`.

Moreover, as for `$gather`, the values of `index` must be between `0`
and `self$size(dim) - 1` inclusive, and all values in a row along the
specified dimension `dim` must be unique.

#### Arguments:

- dim (int): the axis along which to index
- index (LongTensor): the indices of elements to scatter,
- can be either empty or the same size of src. When empty, the operation
  returns identity
- src (Tensor): the source element(s) to scatter,
- incase `value` is not specified
- value (float): the source element(s) to scatter,
- incase `src` is not specified

#### Examples:

``` r
x <- torch_rand(2, 5)
x
torch_zeros(3, 5)$scatter_(
        1, 
        torch_tensor(rbind(c(2, 3, 3, 1, 1), c(3, 1, 1, 2, 3)), x)
)

z <- torch_zeros(2, 4)$scatter_(
        2, 
        torch_tensor(matrix(3:4, ncol = 1)), 1.23
)
```

## scatter_add

scatter_add(dim, index, src) -\> Tensor

Out-of-place version of `$scatter_add_`

## scatter_add\_

scatter_add\_(dim, index, src) -\> Tensor

Adds all values from the tensor `other` into `self` at the indices
specified in the `index` tensor in a similar fashion as `~$scatter_`.
For each value in `src`, it is added to an index in `self` which is
specified by its index in `src` for `dimension != dim` and by the
corresponding value in `index` for `dimension = dim`.

For a 3-D tensor, `self` is updated as::

    self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

`self`, `index` and `src` should have same number of dimensions. It is
also required that `index.size(d) <= src.size(d)` for all dimensions
`d`, and that `index.size(d) <= self$size(d)` for all dimensions
`d != dim`.

#### Note:

In some circumstances when using the CUDA backend with CuDNN, this
operator may select a nondeterministic algorithm to increase
performance. If this is undesirable, you can try to make the operation
deterministic (potentially at a performance cost) by setting
`torch_backends.cudnn.deterministic = TRUE`.

#### Arguments:

- dim (int): the axis along which to index
- index (LongTensor): the indices of elements to scatter and add,
- can be either empty or the same size of src. When empty, the operation
  returns identity.
- src (Tensor): the source elements to scatter and add

#### Examples:

``` r
x <- torch_rand(2, 5)
x
torch_ones(3, 5)$scatter_add_(1, torch_tensor(rbind(c(0, 1, 2, 0, 0), c(2, 0, 0, 1, 2))), x)
```

## select

select(dim, index) -\> Tensor

Slices the `self` tensor along the selected dimension at the given
index. This function returns a view of the original tensor with the
given dimension removed.

#### Arguments:

- dim (int): the dimension to slice
- index (int): the index to select with

#### Note:

`select` is equivalent to slicing. For example,
`tensor$select(0, index)` is equivalent to `tensor[index]` and
`tensor$select(2, index)` is equivalent to `tensor[:,:,index]`.

## set\_

set\_(source=NULL, storage_offset=0, size=NULL, stride=NULL) -\> Tensor

Sets the underlying storage, size, and strides. If `source` is a tensor,
`self` tensor will share the same storage and have the same size and
strides as `source`. Changes to elements in one tensor will be reflected
in the other.

#### Arguments:

- source (Tensor or Storage): the tensor or storage to use
- storage_offset (int, optional): the offset in the storage
- size (torch_Size, optional): the desired size. Defaults to the size of
  the source.
- stride (tuple, optional): the desired stride. Defaults to C-contiguous
  strides.

## share_memory\_

Moves the underlying storage to shared memory.

This is a no-op if the underlying storage is already in shared memory
and for CUDA tensors. Tensors in shared memory cannot be resized.

## short

short(memory_format=torch_preserve_format) -\> Tensor

`self$short()` is equivalent to `self$to(torch_int16)`. See \[to()\].

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of
- returned Tensor. Default: `torch_preserve_format`.

## sigmoid

sigmoid() -\> Tensor

See
[`?torch_sigmoid`](https://torch.mlverse.org/docs/dev/reference/torch_sigmoid.md)

## sigmoid\_

sigmoid\_() -\> Tensor

In-place version of `$sigmoid`

## sign

sign() -\> Tensor

See
[`?torch_sign`](https://torch.mlverse.org/docs/dev/reference/torch_sign.md)

## sign\_

sign\_() -\> Tensor

In-place version of `$sign`

## sin

sin() -\> Tensor

See
[`?torch_sin`](https://torch.mlverse.org/docs/dev/reference/torch_sin.md)

## sin\_

sin\_() -\> Tensor

In-place version of `$sin`

## sinh

sinh() -\> Tensor

See
[`?torch_sinh`](https://torch.mlverse.org/docs/dev/reference/torch_sinh.md)

## sinh\_

sinh\_() -\> Tensor

In-place version of `$sinh`

## size

size() -\> torch_Size

Returns the size of the `self` tensor. The returned value is a subclass
of `tuple`.

#### Examples:

``` r
torch_empty(3, 4, 5)$size()
```

## slogdet

slogdet() -\> (Tensor, Tensor)

See
[`?torch_slogdet`](https://torch.mlverse.org/docs/dev/reference/torch_slogdet.md)

## sort

sort(dim=-1, descending=FALSE) -\> (Tensor, LongTensor)

See
[`?torch_sort`](https://torch.mlverse.org/docs/dev/reference/torch_sort.md)

## sparse_dim

sparse_dim() -\> int

If `self` is a sparse COO tensor (i.e., with `torch_sparse_coo` layout),
this returns the number of sparse dimensions. Otherwise, this throws an
error.

See also `Tensor.dense_dim`.

## sparse_mask

sparse_mask(input, mask) -\> Tensor

Returns a new SparseTensor with values from Tensor `input` filtered by
indices of `mask` and values are ignored. `input` and `mask` must have
the same shape.

#### Arguments:

- input (Tensor): an input Tensor
- mask (SparseTensor): a SparseTensor which we filter `input` based on
  its indices

## split

See
[`?torch_split`](https://torch.mlverse.org/docs/dev/reference/torch_split.md)

## sqrt

sqrt() -\> Tensor

See
[`?torch_sqrt`](https://torch.mlverse.org/docs/dev/reference/torch_sqrt.md)

## sqrt\_

sqrt\_() -\> Tensor

In-place version of `$sqrt`

## square

square() -\> Tensor

See
[`?torch_square`](https://torch.mlverse.org/docs/dev/reference/torch_square.md)

## square\_

square\_() -\> Tensor

In-place version of `$square`

## squeeze

squeeze(dim=NULL) -\> Tensor

See
[`?torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)

## squeeze\_

squeeze\_(dim=NULL) -\> Tensor

In-place version of `$squeeze`

## std

std(dim=NULL, unbiased=TRUE, keepdim=FALSE) -\> Tensor

See
[`?torch_std`](https://torch.mlverse.org/docs/dev/reference/torch_std.md)

## stft

See
[`?torch_stft`](https://torch.mlverse.org/docs/dev/reference/torch_stft.md)

## storage

storage() -\> torch_Storage

Returns the underlying storage.

## storage_offset

storage_offset() -\> int

Returns `self` tensor’s offset in the underlying storage in terms of
number of storage elements (not bytes).

#### Examples:

``` r
x <- torch_tensor(c(1, 2, 3, 4, 5))
x$storage_offset()
x[3:N]$storage_offset()
```

## storage_type

storage_type() -\> type

Returns the type of the underlying storage.

## stride

stride(dim) -\> tuple or int

Returns the stride of `self` tensor.

Stride is the jump necessary to go from one element to the next one in
the specified dimension `dim`. A tuple of all strides is returned when
no argument is passed in. Otherwise, an integer value is returned as the
stride in the particular dimension `dim`.

#### Arguments:

- dim (int, optional): the desired dimension in which stride is required

#### Examples:

``` r
x <- torch_tensor(matrix(1:10, nrow = 2))
x$stride()
x$stride(1)
x$stride(-1)
```

## sub

sub(other, \*, alpha=1) -\> Tensor

Subtracts a scalar or tensor from `self` tensor. If both `alpha` and
`other` are specified, each element of `other` is scaled by `alpha`
before being used.

When `other` is a tensor, the shape of `other` must be
`broadcastable <broadcasting-semantics>` with the shape of the
underlying tensor.

## sub\_

sub\_(other, \*, alpha=1) -\> Tensor

In-place version of `$sub`

## sum

sum(dim=NULL, keepdim=FALSE, dtype=NULL) -\> Tensor

See
[`?torch_sum`](https://torch.mlverse.org/docs/dev/reference/torch_sum.md)

## sum_to_size

sum_to_size(\*size) -\> Tensor

Sum `this` tensor to `size`. `size` must be broadcastable to `this`
tensor size.

#### Arguments:

- size (int…): a sequence of integers defining the shape of the output
  tensor.

## svd

svd(some=TRUE, compute_uv=TRUE) -\> (Tensor, Tensor, Tensor)

See
[`?torch_svd`](https://torch.mlverse.org/docs/dev/reference/torch_svd.md)

## t

t() -\> Tensor

See
[`?torch_t`](https://torch.mlverse.org/docs/dev/reference/torch_t.md)

## t\_

t\_() -\> Tensor

In-place version of `$t`

## take

take(indices) -\> Tensor

See
[`?torch_take`](https://torch.mlverse.org/docs/dev/reference/torch_take.md)

## tan

tan() -\> Tensor

See
[`?torch_tan`](https://torch.mlverse.org/docs/dev/reference/torch_tan.md)

## tan\_

tan\_() -\> Tensor

In-place version of `$tan`

## tanh

tanh() -\> Tensor

See
[`?torch_tanh`](https://torch.mlverse.org/docs/dev/reference/torch_tanh.md)

## tanh\_

tanh\_() -\> Tensor

In-place version of `$tanh`

## to

to(\*args, \*\*kwargs) -\> Tensor

Performs Tensor dtype and/or device conversion. A `torch_dtype` and
:class:`torch_device` are inferred from the arguments of
`self$to(*args, **kwargs)`.

#### Note:

If the `self` Tensor already has the correct `torch_dtype` and
:class:`torch_device`, then `self` is returned. Otherwise, the returned
tensor is a copy of `self` with the desired `torch_dtype` and
:class:`torch_device`.

Here are the ways to call `to`:

to(dtype, non_blocking=FALSE, copy=FALSE,
memory_format=torch_preserve_format) -\> Tensor

Returns a Tensor with the specified `dtype`

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of returned Tensor. Default: `torch_preserve_format`.

to(device=NULL, dtype=NULL, non_blocking=FALSE, copy=FALSE,
memory_format=torch_preserve_format) -\> Tensor

Returns a Tensor with the specified `device` and (optional) `dtype`. If
`dtype` is `NULL` it is inferred to be `self$dtype`. When
`non_blocking`, tries to convert asynchronously with respect to the host
if possible, e.g., converting a CPU Tensor with pinned memory to a CUDA
Tensor.

When `copy` is set, a new Tensor is created even when the Tensor already
matches the desired conversion.

#### Arguments:

- memory_format (`torch_memory_format`, optional): the desired memory
  format of returned Tensor. Default: `torch_preserve_format`.

function:: to(other, non_blocking=FALSE, copy=FALSE) -\> Tensor

Returns a Tensor with same `torch_dtype` and :class:`torch_device` as
the Tensor `other`. When `non_blocking`, tries to convert asynchronously
with respect to the host if possible, e.g., converting a CPU Tensor with
pinned memory to a CUDA Tensor.

When `copy` is set, a new Tensor is created even when the Tensor already
matches the desired conversion.

#### Examples:

``` r
tensor <- torch_randn(2, 2)  # Initially dtype=float32, device=cpu
tensor$to(dtype = torch_float64())

other <- torch_randn(1, dtype=torch_float64())
tensor$to(other = other, non_blocking=TRUE)
```

## to_mkldnn

to_mkldnn() -\> Tensor Returns a copy of the tensor in `torch_mkldnn`
layout.

## to_sparse

to_sparse(sparseDims) -\> Tensor Returns a sparse copy of the tensor.
PyTorch supports sparse tensors in `coordinate format <sparse-docs>`.

#### Arguments:

- sparseDims (int, optional): the number of sparse dimensions to include
  in the new sparse tensor

## tolist

tolist() -\> list or number

Returns the tensor as a (nested) list. For scalars, a standard Python
number is returned, just like with `$item`. Tensors are automatically
moved to the CPU first if necessary.

This operation is not differentiable.

## topk

topk(k, dim=NULL, largest=TRUE, sorted=TRUE) -\> (Tensor, LongTensor)

See
[`?torch_topk`](https://torch.mlverse.org/docs/dev/reference/torch_topk.md)

## trace

trace() -\> Tensor

See
[`?torch_trace`](https://torch.mlverse.org/docs/dev/reference/torch_trace.md)

## transpose

transpose(dim0, dim1) -\> Tensor

See
[`?torch_transpose`](https://torch.mlverse.org/docs/dev/reference/torch_transpose.md)

## transpose\_

transpose\_(dim0, dim1) -\> Tensor

In-place version of `$transpose`

## triangular_solve

triangular_solve(A, upper=TRUE, transpose=FALSE, unitriangular=FALSE)
-\> (Tensor, Tensor)

See \[torch_triangular_solve()\]

## tril

tril(k=0) -\> Tensor

See
[`?torch_tril`](https://torch.mlverse.org/docs/dev/reference/torch_tril.md)

## tril\_

tril\_(k=0) -\> Tensor

In-place version of `$tril`

## triu

triu(k=0) -\> Tensor

See
[`?torch_triu`](https://torch.mlverse.org/docs/dev/reference/torch_triu.md)

## triu\_

triu\_(k=0) -\> Tensor

In-place version of `$triu`

## true_divide

true_divide(value) -\> Tensor

See \[torch_true_divide()\]

## true_divide\_

true_divide\_(value) -\> Tensor

In-place version of `$true_divide_`

## trunc

trunc() -\> Tensor

See
[`?torch_trunc`](https://torch.mlverse.org/docs/dev/reference/torch_trunc.md)

## trunc\_

trunc\_() -\> Tensor

In-place version of `$trunc`

## type

type(dtype=NULL, non_blocking=FALSE, \*\*kwargs) -\> str or Tensor
Returns the type if `dtype` is not provided, else casts this object to
the specified type.

If this is already of the correct type, no copy is performed and the
original object is returned.

#### Arguments:

- dtype (type or string): The desired type
- non_blocking (bool): If `TRUE`, and the source is in pinned memory
- and destination is on the GPU or vice versa, the copy is performed
- asynchronously with respect to the host. Otherwise, the argument
- has no effect. \*\*kwargs: For compatibility, may contain the key
  `async` in place of
- the `non_blocking` argument. The `async` arg is deprecated.

## type_as

type_as(tensor) -\> Tensor

Returns this tensor cast to the type of the given tensor.

This is a no-op if the tensor is already of the correct type. This is
equivalent to `self$type(tensor.type())`

#### Arguments:

- tensor (Tensor): the tensor which has the desired type

## unbind

unbind(dim=0) -\> seq

See
[`?torch_unbind`](https://torch.mlverse.org/docs/dev/reference/torch_unbind.md)

## unflatten

Unflattens the named dimension `dim`, viewing it in the shape specified
by `namedshape`.

#### Arguments:

- namedshape: (iterable of `(name, size)` tuples).

## unfold

unfold(dimension, size, step) -\> Tensor

Returns a view of the original tensor which contains all slices of size
`size` from `self` tensor in the dimension `dimension`.

Step between two slices is given by `step`.

If `sizedim` is the size of dimension `dimension` for `self`, the size
of dimension `dimension` in the returned tensor will be
`(sizedim - size) / step + 1`.

An additional dimension of size `size` is appended in the returned
tensor.

#### Arguments:

- dimension (int): dimension in which unfolding happens
- size (int): the size of each slice that is unfolded
- step (int): the step between each slice

## uniform\_

uniform\_(from=0, to=1) -\> Tensor

Fills `self` tensor with numbers sampled from the continuous uniform
distribution:

\\ P(x) = \dfrac{1}{\text{to} - \text{from}} \\

## unique

Returns the unique elements of the input tensor.

See `?torch_unique`

## unique_consecutive

Eliminates all but the first element from every consecutive group of
equivalent elements.

See \[torch_unique_consecutive()\]

## unsqueeze

unsqueeze(dim) -\> Tensor

See
[`?torch_unsqueeze`](https://torch.mlverse.org/docs/dev/reference/torch_unsqueeze.md)

## unsqueeze\_

unsqueeze\_(dim) -\> Tensor

In-place version of `$unsqueeze`

## values

values() -\> Tensor

If `self` is a sparse COO tensor (i.e., with `torch_sparse_coo` layout),
this returns a view of the contained values tensor. Otherwise, this
throws an error.

#### Note:

This method can only be called on a coalesced sparse tensor. See
`Tensor$coalesce` for details.

## var

var(dim=NULL, unbiased=TRUE, keepdim=FALSE) -\> Tensor

See
[`?torch_var`](https://torch.mlverse.org/docs/dev/reference/torch_var.md)

## view

view(\*shape) -\> Tensor

Returns a new tensor with the same data as the `self` tensor but of a
different `shape`.

The returned tensor shares the same data and must have the same number
of elements, but may have a different size. For a tensor to be viewed,
the new view size must be compatible with its original size and stride,
i.e., each new view dimension must either be a subspace of an original
dimension, or only span across original dimensions `d, d+1, \dots, d+k`
that satisfy the following contiguity-like condition that
`\forall i = d, \dots, d+k-1`,

\\ \text{stride}\[i\] = \text{stride}\[i+1\] \times \text{size}\[i+1\]
\\

Otherwise, it will not be possible to view `self` tensor as `shape`
without copying it (e.g., via `contiguous`). When it is unclear whether
a `view` can be performed, it is advisable to use :meth:`reshape`, which
returns a view if the shapes are compatible, and copies (equivalent to
calling `contiguous`) otherwise.

#### Arguments:

- shape (torch_Size or int…): the desired size

## view_as

view_as(other) -\> Tensor

View this tensor as the same size as `other`. `self$view_as(other)` is
equivalent to `self$view(other.size())`.

Please see `$view` for more information about `view`.

#### Arguments:

- other (\`\$): The result tensor has the same size
- as `other`.

## where

where(condition, y) -\> Tensor

`self$where(condition, y)` is equivalent to
`torch_where(condition, self, y)`. See
[`?torch_where`](https://torch.mlverse.org/docs/dev/reference/torch_where.md)

## zero\_

zero\_() -\> Tensor

Fills `self` tensor with zeros.
