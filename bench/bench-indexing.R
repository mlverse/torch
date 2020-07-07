Sys.setenv(KMP_DUPLICATE_LIB_OK=TRUE)
x <- torch_randn(1000, 10, 10)

bench::mark(min_time = 1,
  x[1,,],
  torch_select(x, 0, 1)
)


profvis::profvis({
  for (i in 1:500)
    x[1,,]
})
