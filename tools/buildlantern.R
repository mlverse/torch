cat("Starting Lantern build process...\n")

if (dir.exists("src/lantern")) {
  cat("Lantern directory exists. Proceeding...\n")
  
  # Ensure the build directory is created
  if (!dir.exists("src/lantern/build")) {
    cat("Creating build directory...\n")
    dir.create("src/lantern/build", recursive = TRUE)
    cat("Build directory created.\n")
  } else {
    cat("Build directory already exists.\n")
  }
  
  # Run CMake commands
  withr::with_dir("src/lantern/build", {
    cat("Running CMake configuration with MinGW...\n")
    cmake_result <- system(
      "cmake -G \"MinGW Makefiles\" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release ..",
      intern = TRUE
    )
    cat("CMake configuration output:\n", paste(cmake_result, collapse = "\n"), "\n")
    
    cat("Running make with MinGW...\n")
    make_result <- system("mingw32-make VERBOSE=1", intern = TRUE)
    cat("Make output:\n", paste(make_result, collapse = "\n"), "\n")
  })
  
  cat("Lantern build process completed successfully.\n")
} else {
  cat("Lantern directory does not exist. Please check the path.\n")
}
