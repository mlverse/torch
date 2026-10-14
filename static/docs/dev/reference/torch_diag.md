# Diag

Diag

## Usage

``` r
torch_diag(self, diagonal = 0L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- diagonal:

  (int, optional) the diagonal to consider

## diag(input, diagonal=0, out=NULL) -\> Tensor

- If `input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of `input` as the diagonal.

- If `input` is a matrix (2-D tensor), then returns a 1-D tensor with
  the diagonal elements of `input`.

The argument `diagonal` controls which diagonal to consider:

- If `diagonal` = 0, it is the main diagonal.

- If `diagonal` \> 0, it is above the main diagonal.

- If `diagonal` \< 0, it is below the main diagonal.
