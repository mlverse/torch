library(torch)

p = 100
steps = 1000
n = 1000

device = "mps"

X = torch_randn(n, p, device = device)
beta = torch_randn(p, 1, device = device)
Y = X$matmul(beta)

latent = 5000

net = nn_sequential(
  nn_linear(p, latent),
  nn_relu(),
  nn_linear(latent, 1)
)

net$to(device = device)
opt = optim_adam(net$parameters, lr = 0.01)

t1 = Sys.time()

for (i in 1:steps) {
  opt$zero_grad(set_to_none = TRUE)
  Y_hat = net(X)
  loss = nnf_mse_loss(Y, Y_hat)
  loss$backward()
}

t2 = Sys.time()

print(paste0("Total time: ", t2 - t1))