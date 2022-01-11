library(torch)

# creates example tensors. x requires_grad = TRUE tells that
# we are going to take derivatives over it.
x <- torch_tensor(3, requires_grad = TRUE)
y <- torch_tensor(2)

# executes the forward operation x^2
o <- x^2

# computes the backward operation for each tensor that is marked with
# requires_grad = TRUE
o$backward()

# get do/dx = 2 * x (at x = 3)
x$grad
