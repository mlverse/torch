# One_hot

Takes LongTensor with index values of shape `(*)` and returns a tensor
of shape `(*, num_classes)` that have zeros everywhere except where the
index of last dimension matches the corresponding value of the input
tensor, in which case it will be 1.

## Usage

``` r
nnf_one_hot(tensor, num_classes = -1)
```

## Arguments

- tensor:

  (LongTensor) class values of any shape.

- num_classes:

  (int) Total number of classes. If set to -1, the number of classes
  will be inferred as one greater than the largest class value in the
  input tensor.

## Details

One-hot on Wikipedia: https://en.wikipedia.org/wiki/One-hot
