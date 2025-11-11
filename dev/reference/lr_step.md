# Step learning rate decay

Decays the learning rate of each parameter group by gamma every
step_size epochs. Notice that such decay can happen simultaneously with
other changes to the learning rate from outside this scheduler. When
last_epoch=-1, sets initial lr as lr.

## Usage

``` r
lr_step(optimizer, step_size, gamma = 0.1, last_epoch = -1)
```

## Arguments

- optimizer:

  (Optimizer): Wrapped optimizer.

- step_size:

  (int): Period of learning rate decay.

- gamma:

  (float): Multiplicative factor of learning rate decay. Default: 0.1.

- last_epoch:

  (int): The index of last epoch. Default: -1.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
scheduler <- lr_step(optimizer, step_size = 30, gamma = 0.1)
for (epoch in 1:100) {
  train(...)
  validate(...)
  scheduler$step()
}
} # }

}
```
