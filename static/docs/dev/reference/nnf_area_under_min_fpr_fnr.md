# Area under the \\Min(FPR, FNR)\\ (AUM)

Function that measures Area under the \\Min(FPR, FNR)\\ (AUM) between
each element in the \\input\\ and \\target\\.

## Usage

``` r
nnf_area_under_min_fpr_fnr(input, target)
```

## Arguments

- input:

  Tensor of arbitrary shape

- target:

  Tensor of the same shape as input. Should be the factor level of the
  binary outcome, i.e. with values `1L` and `2L`.

## Details

This is used for measuring the error of a binary reconstruction within
highly unbalanced dataset, where the goal is optimizing the ROC curve.
