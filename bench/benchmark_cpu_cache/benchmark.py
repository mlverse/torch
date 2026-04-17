import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

print(f"PyTorch version: {torch.__version__}")

p = 100
steps = 1000
n = 1000
latent = 5000
nreps = 5

def se(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))

results = []
for i in range(nreps):
    X = torch.randn(n, p)
    Y = torch.randn(n, 1)

    net = nn.Sequential(
        nn.Linear(p, latent),
        nn.ReLU(),
        nn.Linear(latent, 1)
    )
    opt = optim.Adam(net.parameters(), lr=0.01)

    t1 = time.time()
    for j in range(steps):
        opt.zero_grad(set_to_none=True)
        Y_hat = net(X)
        loss = nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        opt.step()
    loss.item()
    elapsed = time.time() - t1
    results.append(elapsed)
    print(f"Rep {i+1}: {elapsed:.2f}s")

mean = sum(results) / len(results)
print(f"\nmean={mean:.3f}s  se={se(results):.3f}s")
