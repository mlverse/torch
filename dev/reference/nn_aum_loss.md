# AUM loss

Creates a criterion that measures the Area under the \\Min(FPR, FNR)\\
(AUM) between each element in the input \\pred_tensor\\ and target
\\label_tensor\\.

## Usage

``` r
nn_aum_loss()
```

## Details

This is used for measuring the error of a binary reconstruction within
highly unbalanced dataset, where the goal is optimizing the ROC curve.
Note that the targets \\label_tensor\\ should be factor level of the
binary outcome, i.e. with values `1L` and `2L`.

## References

J. Hillman, T.D. Hocking: Optimizing ROC Curves with a Sort-Based
Surrogate Loss for Binary Classification and Changepoint Detection
https://jmlr.org/papers/volume24/21-0751/21-0751.pdf

## Examples

``` r
if (torch_is_installed()) {
loss <- nn_aum_loss()
input <- torch_randn(4, 6, requires_grad = TRUE)
target <- input > 1.5
output <- loss(input, target)
output$backward()
}
```
