# Compare installed torch builds with: Rscript bench/bench-index-conversion.R
library(torch)

index <- torch_tensor(c(-2, -1, 1, 2), dtype = torch_long())
convert <- torch:::to_index_tensor
for (i in seq_len(100)) invisible(convert(index))

for (i in seq_len(4)) {
  elapsed <- system.time(for (j in seq_len(10000)) invisible(convert(index)))[["elapsed"]]
  cat(sprintf("trial=%d elapsed=%.3f seconds\n", i, elapsed))
}
