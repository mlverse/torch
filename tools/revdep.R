
revdepcheck::revdep_check(num_workers = 4, pkg = ".",
                          env = c(revdepcheck::revdep_env_vars(),
                                  TORCH_INSTALL = "1",
                                  TORCH_TEST = "1"))

