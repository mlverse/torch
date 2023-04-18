test_that("cpp_jit_all_operators() works", {
  ops <- cpp_jit_all_operators()
  expect_type(ops, "character")
})

test_that("cpp_jit_operator_info() works", {
  info <- cpp_jit_operator_info("aten::matmul")
  expect_length(info$arguments, 2)
  expect_equal(info$arguments$self, "TensorType")
  expect_length(info$returns, 1)
  expect_equal(info$returns[[1]], "TensorType")
})

test_that("cpp_jit_execute() works", {
 
  # matmul, default use
  res <- cpp_jit_execute("aten::matmul", list(torch::torch_ones(5, 4), torch::torch_rand(4, 5)))
  expect_equal(length(res), 1)
  expect_equal(res[[1]] |> dim(), c(5, 5))
  
  # matmul, C10 error: mat1 and mat2 shapes cannot be multiplied (4x5 and 4x5)
  # currently we return the original Stack (maybe change that logic)
  stack <- list(torch::torch_ones(4, 5), torch::torch_rand(4, 5))
  res <- cpp_jit_execute("aten::matmul", stack)
  expect_equal_to_tensor(stack[[1]], res[[1]])
  expect_equal_to_tensor(stack[[2]], res[[2]])
  
  # matmul, C10 error: Expected Tensor but got GenericList
  # currently we return the original Stack (maybe change that logic)
  stack <- list(torch::torch_ones(4, 5), 27)
  res <- cpp_jit_execute("aten::matmul", stack)
  expect_equal(stack[[1]], res[[1]])
  expect_equal(stack[[2]], res[[2]])
  
  # matmul, passing out tensor
  res <- cpp_jit_execute("aten::matmul", list(torch::torch_ones(5, 4), torch::torch_rand(4, 5), torch::torch_zeros(5, 5)))
  expect_equal(length(res), 1)
  expect_equal(res[[1]] |> dim(), c(5, 5))
  
})

test_that("cpp_jit_all_schemas_for() works", {
  res <- cpp_jit_all_schemas_for("aten::conv2d")
  expect_equal(setdiff(unname(unlist(res[[2]]$arguments)), unname(unlist(res[[1]]$arguments))), "StringType")
  res <- cpp_jit_all_schemas_for("aten::matmul")
  expect_equal(length(res), 2)
})

# test_that("explore existing overloads") {
#   res <- cpp_jit_all_operators()
#   num_schemas <- table(res) |> as.data.frame()
#   multiplicities <- num_schemas |> dplyr::group_by(Freq) |> dplyr::summarise(count  = dplyr::n())
#   # Freq count
#   # 1     1   833
#   # 2     2   772
#   # 3     3    44
#   # 4     4   102
#   # 5     5    24
#   # 6     6    64
#   # 7     7    12
#   # 8     8     8
#   # 9     9     9
#   # 10    10     8
#   # 11    11     3
#   # 12    12     3
#   # 13    13     2
#   # 14    14     2
#   # 15    15     1
#   # 16    16     3
#   # 17    17     1
#   # 18    21     2
#   twofold <- num_schemas |> dplyr::filter(Freq == 2, grepl("aten::[^_]", res)) |> head(3)
#   # out tensor (increments arguments as well as returns by 1) (aten::abs, aten::absolute, aten::adaptive_avg_pool2d)
#   for (i in 1:3) {
#     cat("Operator: ", twofold$res[i] |> as.character(), "\n")
#     schemas <- cpp_jit_all_schemas_for(twofold$res[i] |> as.character())
#     print(schemas)
#     cat("\n")
#   }
#   threefold <- num_schemas |> dplyr::filter(Freq == 3, grepl("aten::[^_]", res)) |> head(3)
#   # different argument and/or return types (aten::add_, aten::Bool, aten::degrees)
#   for (i in 1:3) {
#     cat("Operator: ", threefold$res[i] |> as.character(), "\n")
#     schemas <- cpp_jit_all_schemas_for(threefold$res[i] |> as.character())
#     print(schemas)
#     cat("\n")
#   }
#   fourfold <- num_schemas |> dplyr::filter(Freq == 4, grepl("aten::[^_]", res)) |> head(3)
#   # different argument and/or return types (aten::argsort, aten::bartlett_window, aten::blackman_window)
#   for (i in 1:3) {
#     cat("Operator: ", fourfold$res[i] |> as.character(), "\n")
#     schemas <- cpp_jit_all_schemas_for(fourfold$res[i] |> as.character())
#     print(schemas)
#     cat("\n")
#   }
# }

