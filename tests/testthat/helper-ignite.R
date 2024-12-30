make_optimizer_maker = function(optimizer_fn) {
  function(..., steps = 2) {
    n <- nn_linear(1, 1)
    o <- optimizer_fn(n$parameters, ...)
    x <- torch_randn(10, 1)
    y <- torch_randn(10, 1)
    s <- function() {
      o$zero_grad()
      loss <- mean((n(x) - y)^2)
      loss$backward()
      o$step()
    }
    replicate(steps, s())
    o
  }
}

make_ignite_adamw <- make_optimizer_maker(optim_ignite_adamw)
make_ignite_sgd <- make_optimizer_maker(optim_ignite_sgd)
make_ignite_adam <- make_optimizer_maker(optim_ignite_adam)
make_ignite_rmsprop <- make_optimizer_maker(optim_ignite_rmsprop)
make_ignite_adagrad <- make_optimizer_maker(optim_ignite_adagrad)
