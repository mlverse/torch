# Indexing tensors

``` r
library(torch)
```

In this article we describe the indexing operator for torch tensors and
how it compares to the R indexing operator for arrays.

Torch’s indexing semantics are closer to numpy’s semantics than R’s. You
will find a lot of similarities between this article and the `numpy`
indexing article available
[here](https://docs.scipy.org/doc/numpy-1.10.0/user/basics.indexing.html).

## Single element indexing

Single element indexing for a 1-D tensors works mostly as expected. Like
R, it is 1-based. Unlike R though, it accepts negative indices for
indexing from the end of the array. (In R, negative indices are used to
remove elements.)

``` r
x <- torch_tensor(1:10)
x[1]
#> torch_tensor
#> 1
#> [ CPULongType{} ]
x[-1]
#> torch_tensor
#> 10
#> [ CPULongType{} ]
```

You can also subset matrices and higher dimensions arrays using the same
syntax:

``` r
x <- x$reshape(shape = c(2,5))
x
#> torch_tensor
#>   1   2   3   4   5
#>   6   7   8   9  10
#> [ CPULongType{2,5} ]
x[1,3]
#> torch_tensor
#> 3
#> [ CPULongType{} ]
x[1,-1]
#> torch_tensor
#> 5
#> [ CPULongType{} ]
```

Note that if one indexes a multidimensional tensor with fewer indices
than dimensions, torch’s behaviour differs from R, which flattens the
array. In torch, the missing indices are considered complete slices `:`.

``` r
x[1]
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPULongType{5} ]
```

## Slicing and striding

It is possible to slice and stride arrays to extract sub-arrays of the
same number of dimensions, but of different sizes than the original.
This is best illustrated by a few examples:

``` r
x <- torch_tensor(1:10)
x
#> torch_tensor
#>   1
#>   2
#>   3
#>   4
#>   5
#>   6
#>   7
#>   8
#>   9
#>  10
#> [ CPULongType{10} ]
x[2:5]
#> torch_tensor
#>  2
#>  3
#>  4
#>  5
#> [ CPULongType{4} ]
x[1:(-7)]
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CPULongType{4} ]
```

You can also use the `1:10:2` syntax which means: In the range from 1 to
10, take every second item. For example:

``` r
x[1:5:2]
#> torch_tensor
#>  1
#>  3
#>  5
#> [ CPULongType{3} ]
```

Another special syntax is the `N`, meaning the size of the specified
dimension.

``` r
x[5:N]
#> torch_tensor
#>   5
#>   6
#>   7
#>   8
#>   9
#>  10
#> [ CPULongType{6} ]
```

> Note: the slicing behavior relies on [Non Standard
> Evaluation](https://adv-r.hadley.nz/evaluation.html#evaluation). It
> requires that the expression is passed to the `[` not exactly the
> resulting R vector.

To allow dynamic dynamic indices, you can create a new slice using the
`slc` function. For example:

``` r
x[1:5:2]
#> torch_tensor
#>  1
#>  3
#>  5
#> [ CPULongType{3} ]
```

is equivalent to:

``` r
x[slc(start = 1, end = 5, step = 2)]
#> torch_tensor
#>  1
#>  3
#>  5
#> [ CPULongType{3} ]
```

## Getting the complete dimension

Like in R, you can take all elements in a dimension by leaving an index
empty.

Consider a matrix:

``` r
x <- torch_randn(2, 3)
x
#> torch_tensor
#> -0.4563  1.5989 -0.2974
#>  0.1909  0.8735  0.2792
#> [ CPUFloatType{2,3} ]
```

The following syntax will give you the first row:

``` r
x[1,]
#> torch_tensor
#> -0.4563
#>  1.5989
#> -0.2974
#> [ CPUFloatType{3} ]
```

And this would give you the first 2 columns:

``` r
x[,1:2]
#> torch_tensor
#> -0.4563  1.5989
#>  0.1909  0.8735
#> [ CPUFloatType{2,2} ]
```

## Dropping dimensions

By default, when indexing by a single integer, this dimension will be
dropped to avoid the singleton dimension:

``` r
x <- torch_randn(2, 3)
x[1,]$shape
#> [1] 3
```

You can optionally use the `drop = FALSE` argument to avoid dropping the
dimension.

``` r
x[1,,drop = FALSE]$shape
#> [1] 1 3
```

## Adding a new dimension

It’s possible to add a new dimension to a tensor using index-like
syntax:

``` r
x <- torch_tensor(c(10))
x$shape
#> [1] 1
x[, newaxis]$shape
#> [1] 1 1
x[, newaxis, newaxis]$shape
#> [1] 1 1 1
```

You can also use `NULL` instead of `newaxis`:

``` r
x[,NULL]$shape
#> [1] 1 1
```

## Dealing with variable number of indices

Sometimes we don’t know how many dimensions a tensor has, but we do know
what to do with the last available dimension, or the first one. To
subsume all others, we can use `..`:

``` r
z <- torch_tensor(1:125)$reshape(c(5,5,5))
z[1,..]
#> torch_tensor
#>   1   2   3   4   5
#>   6   7   8   9  10
#>  11  12  13  14  15
#>  16  17  18  19  20
#>  21  22  23  24  25
#> [ CPULongType{5,5} ]
z[..,1]
#> torch_tensor
#>    1    6   11   16   21
#>   26   31   36   41   46
#>   51   56   61   66   71
#>   76   81   86   91   96
#>  101  106  111  116  121
#> [ CPULongType{5,5} ]
```

## Indexing with vectors

Vector indexing is also supported but care must be taken regarding
performance as, in general its much less performant than slice based
indexing.

> Note: Starting from version 0.5.0, vector indexing in torch follows R
> semantics, prior to that the behavior was similar to [numpy’s advanced
> indexing](https://numpy.org/doc/2.2/user/basics.indexing.html#advanced-indexing).
> To use the old behavior, consider using
> [`?torch_index`](https://torch.mlverse.org/docs/dev/reference/torch_index.md),
> [`?torch_index_put`](https://torch.mlverse.org/docs/dev/reference/torch_index_put.md)
> or `torch_index_put_`.

``` r
x <- torch_randn(4,4)
x[c(1,3), c(1,3)]
#> torch_tensor
#> -0.4621  2.0902
#>  0.5317  3.0415
#> [ CPUFloatType{2,2} ]
```

You can also use boolean vectors, for example:

``` r
x[c(TRUE, FALSE, TRUE, FALSE), c(TRUE, FALSE, TRUE, FALSE)]
#> torch_tensor
#> -0.4621  2.0902
#>  0.5317  3.0415
#> [ CPUFloatType{2,2} ]
```

The above examples also work if the index were long or boolean tensors,
instead of R vectors. It’s also possible to index with multi-dimensional
boolean tensors:

``` r
x <- torch_tensor(rbind(
  c(1,2,3),
  c(4,5,6)
))
x[x>3]
#> torch_tensor
#>  4
#>  5
#>  6
#> [ CPUFloatType{3} ]
```
