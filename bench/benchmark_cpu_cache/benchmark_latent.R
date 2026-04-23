library(ggplot2)

DIR <- normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
benchmark_script <- file.path(DIR, "benchmark.R")
benchmark_py <- file.path(DIR, "benchmark.py")

latent_sizes <- c(10, 50, 100, 500, 1000, 2000, 5000)

# Parse output like "mean=4.214s  se=0.065s"
parse_output <- function(lines) {
  mean_line <- grep("^mean=", lines, value = TRUE)
  m <- regmatches(mean_line, regexec("mean=([0-9.]+)s\\s+se=([0-9.]+)s", mean_line))[[1]]
  list(mean = as.numeric(m[2]), se = as.numeric(m[3]))
}

# Check if uv is available for running Python benchmarks
python_available <- tryCatch({
  out <- system2("uv", c("run", "--with", "torch==2.8.0", "--with", "numpy",
                          "python", "-c", "import torch; print(torch.__version__)"),
                 stdout = TRUE, stderr = TRUE)
  length(out) > 0 && !any(grepl("Error|error|No module", out))
}, error = function(e) FALSE)

results <- data.frame()

for (latent in latent_sizes) {
  for (cache in c(TRUE, FALSE)) {
    cache_arg <- if (!cache) "--no-cache" else ""
    cat(sprintf("Running: latent=%d cache=%s\n", latent, if (cache) "enabled" else "disabled"))

    out <- system2("Rscript", c(benchmark_script, sprintf("--latent=%d", latent), cache_arg),
                   stdout = TRUE, stderr = TRUE)
    cat(out, sep = "\n")

    parsed <- parse_output(out)
    results <- rbind(results, data.frame(
      latent = latent,
      mode = if (cache) "cache enabled" else "cache disabled",
      mean_s = parsed$mean,
      se_s = parsed$se,
      stringsAsFactors = FALSE
    ))
  }

  if (python_available) {
    cat(sprintf("Running: latent=%d Python\n", latent))
    out <- system2("uv", c("run", "--with", "torch==2.8.0", "--with", "numpy",
                            "python", benchmark_py, sprintf("--latent=%d", latent)),
                   stdout = TRUE, stderr = TRUE)
    cat(out, sep = "\n")

    parsed <- parse_output(out)
    results <- rbind(results, data.frame(
      latent = latent,
      mode = "Python PyTorch",
      mean_s = parsed$mean,
      se_s = parsed$se,
      stringsAsFactors = FALSE
    ))
  }
}

# Print table
cat("\n=== Results ===\n")
has_python <- "Python PyTorch" %in% results$mode
header <- sprintf("%-8s %16s %16s %8s", "latent", "cache enabled", "cache disabled", "speedup")
if (has_python) header <- paste0(header, sprintf(" %16s %8s", "Python", "R/Python"))
cat(header, "\n")

for (latent in latent_sizes) {
  enabled <- results$mean_s[results$latent == latent & results$mode == "cache enabled"]
  disabled <- results$mean_s[results$latent == latent & results$mode == "cache disabled"]
  line <- sprintf("%-8d %13.3f s %13.3f s %7.2fx", latent, enabled, disabled, disabled / enabled)
  if (has_python) {
    python_t <- results$mean_s[results$latent == latent & results$mode == "Python PyTorch"]
    if (length(python_t) > 0) {
      line <- paste0(line, sprintf(" %13.3f s %7.2fx", python_t, enabled / python_t))
    }
  }
  cat(line, "\n")
}

# Plot
results$latent <- factor(results$latent)

p <- ggplot(results, aes(x = latent, y = mean_s, fill = mode)) +
  geom_col(position = "dodge") +
  geom_errorbar(aes(ymin = mean_s - se_s, ymax = mean_s + se_s),
                position = position_dodge(width = 0.9), width = 0.3) +
  labs(
    title = "Training loop time by hidden layer size",
    subtitle = "2-layer NN (100 -> latent -> 1), Adam, 1000 steps, CPU.",
    x = "Hidden layer size",
    y = "Total time (s)",
    fill = ""
  ) +
  theme_minimal()

outfile <- file.path(DIR, "benchmark_latent.png")
ggsave(outfile, p, width = 10, height = 6, dpi = 150)
cat(sprintf("\nPlot saved to %s\n", outfile))
