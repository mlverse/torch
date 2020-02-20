lantern_sync <- function() {
  if (!dir.exists("src/lantern")) dir.create("src/lantern")
  file.copy(dir("../lantern/include/lantern/", full.names = TRUE), "src/lantern/", overwrite = TRUE)
}
