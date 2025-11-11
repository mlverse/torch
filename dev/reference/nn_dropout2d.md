# Dropout2D module

Randomly zero out entire channels (a channel is a 2D feature map, e.g.,
the \\j\\-th channel of the \\i\\-th sample in the batched input is a 2D
tensor \\\mbox{input}\[i, j\]\\).

## Usage

``` r
nn_dropout2d(p = 0.5, inplace = FALSE)
```

## Arguments

- p:

  (float, optional): probability of an element to be zero-ed.

- inplace:

  (bool, optional): If set to `TRUE`, will do this operation in-place

## Details

Each channel will be zeroed out independently on every forward call with
probability `p` using samples from a Bernoulli distribution. Usually the
input comes from
[nn_conv2d](https://torch.mlverse.org/docs/dev/reference/nn_conv2d.md)
modules.

As described in the paper [Efficient Object Localization Using
Convolutional Networks](https://arxiv.org/abs/1411.4280) , if adjacent
pixels within feature maps are strongly correlated (as is normally the
case in early convolution layers) then i.i.d. dropout will not
regularize the activations and will otherwise just result in an
effective learning rate decrease. In this case, nn_dropout2d will help
promote independence between feature maps and should be used instead.

## Shape

- Input: \\(N, C, H, W)\\

- Output: \\(N, C, H, W)\\ (same shape as input)

## Examples

``` r
if (torch_is_installed()) {
m <- nn_dropout2d(p = 0.2)
input <- torch_randn(20, 16, 32, 32)
output <- m(input)
}
```
