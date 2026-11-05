# Benchmark GPT-2 forward pass to measure end-to-end dispatch overhead.
#
# Compare CRAN vs dev torch by installing each and running:
#   TORCH_INSTALL=1 Rscript bench/bench-gpt2.R
#
# Results on Apple M3, CPU only (CRAN 0.17.0 vs dev 0.17.0.9000):
#
#   | model              | CRAN 0.17.0 | dev 0.17.0.9000 | speedup |
#   |--------------------|-------------|-----------------|---------|
#   | Tiny  (1L, 64-dim) |    1.42ms   |    1.15ms       |   19%   |
#   | Small (2L, 128-dim)|    2.85ms   |    2.48ms       |   13%   |

library(torch)
library(minhub)

cat("torch version:", as.character(packageVersion("torch")), "\n\n")

# --- Tiny GPT-2: dispatch-sensitive ---
cat("=== Tiny GPT-2 (1 layer, 64-dim, seq_len=4) ===\n")
model_tiny <- gpt2(
  vocab_size = 64, n_embd = 64, n_head = 2,
  n_layer = 1, max_pos = 16, pdrop = 0
)
x_tiny <- torch_randint(1, 64, c(1, 4), dtype = torch_long())

with_no_grad({ for (i in 1:5) model_tiny(x_tiny) })
result_tiny <- with_no_grad({
  bench::mark(gpt2_tiny = model_tiny(x_tiny), min_iterations = 500, check = FALSE)
})
print(result_tiny[, c("expression", "min", "median", "itr/sec", "n_itr")])

# --- Small GPT-2: compute-dominated ---
cat("\n=== Small GPT-2 (2 layers, 128-dim, seq_len=32) ===\n")
model_small <- gpt2(
  vocab_size = 256, n_embd = 128, n_head = 4,
  n_layer = 2, max_pos = 64, pdrop = 0
)
x_small <- torch_randint(1, 256, c(1, 32), dtype = torch_long())

with_no_grad({ for (i in 1:5) model_small(x_small) })
result_small <- with_no_grad({
  bench::mark(gpt2_small = model_small(x_small), min_iterations = 200, check = FALSE)
})
print(result_small[, c("expression", "min", "median", "itr/sec", "n_itr")])
