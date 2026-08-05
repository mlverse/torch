# Reduce learning rate on plateau

Reduce learning rate when a metric has stopped improving. Models often
benefit from reducing the learning rate by a factor of 2-10 once
learning stagnates. This scheduler reads a metrics quantity and if no
improvement is seen for a 'patience' number of epochs, the learning rate
is reduced.

## Usage

``` r
lr_reduce_on_plateau(
  optimizer,
  mode = "min",
  factor = 0.1,
  patience = 10,
  threshold = 1e-04,
  threshold_mode = "rel",
  cooldown = 0,
  min_lr = 0,
  eps = 1e-08,
  verbose = FALSE
)
```

## Arguments

- optimizer:

  (Optimizer): Wrapped optimizer.

- mode:

  (str): One of `min`, `max`. In `min` mode, lr will be reduced when the
  quantity monitored has stopped decreasing; in `max` mode it will be
  reduced when the quantity monitored has stopped increasing. Default:
  'min'.

- factor:

  (float): Factor by which the learning rate will be reduced. new_lr \<-
  lr \* factor. Default: 0.1.

- patience:

  (int): Number of epochs with no improvement after which learning rate
  will be reduced. For example, if `patience = 2`, then we will ignore
  the first 2 epochs with no improvement, and will only decrease the LR
  after the 3rd epoch if the loss still hasn't improved then. Default:
  10.

- threshold:

  (float):Threshold for measuring the new optimum, to only focus on
  significant changes. Default: 1e-4.

- threshold_mode:

  (str): One of `rel`, `abs`. In `rel` mode, dynamic_threshold \<- best
  \* ( 1 + threshold ) in 'max' mode or best \* ( 1 - threshold ) in
  `min` mode. In `abs` mode, dynamic_threshold \<- best + threshold in
  `max` mode or best - threshold in `min` mode. Default: 'rel'.

- cooldown:

  (int): Number of epochs to wait before resuming normal operation after
  lr has been reduced. Default: 0.

- min_lr:

  (float or list): A scalar or a list of scalars. A lower bound on the
  learning rate of all param groups or each group respectively. Default:
  0.

- eps:

  (float): Minimal decay applied to lr. If the difference between new
  and old lr is smaller than eps, the update is ignored. Default: 1e-8.

- verbose:

  (bool): If `TRUE`, prints a message to stdout for each update.
  Default: `FALSE`.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{ 
optimizer <- optim_sgd(model$parameters(), lr=0.1, momentum=0.9)
scheduler <- lr_reduce_on_plateau(optimizer, 'min')
for (epoch in 1:10) {
 train(...)
 val_loss <- validate(...)
 # note that step should be called after validate
 scheduler$step(val_loss)
}
} # }
}
```
