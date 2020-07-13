#' @include utils-data.R
NULL

#' MNIST dataset
#' 
#' Prepares the MNIST dataset and optionally downloads it.
#' 
#' @param root (string): Root directory of dataset where ``MNIST/processed/training.pt``
#'   and  ``MNIST/processed/test.pt`` exist.
#' @param train (bool, optional): If True, creates dataset from ``training.pt``,
#'   otherwise from ``test.pt``.
#' @param download (bool, optional): If true, downloads the dataset from the internet and
#'   puts it in root directory. If dataset is already downloaded, it is not
#'   downloaded again.
#' @param transform (callable, optional): A function/transform that  takes in an PIL image
#'   and returns a transformed version. E.g, ``transforms.RandomCrop``
#' @param target_transform (callable, optional): A function/transform that takes in the
#'   target and transforms it.
#'
#' @export
mnist_dataset <- dataset(
  name = "mnist",
  resources = list(
    c("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    c("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    c("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    c("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
  ),
  training_file = 'training.rds',
  test_file = 'test.rds',
  classes = c('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
             '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine'),
  initialize = function(root, train = TRUE, transform = NULL, target_transform = NULL,
                        download = FALSE) {
    
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    
    self$train <- train
    
    if (download)
      self$download()
    
    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    
    if (self$train)
      data_file <- self$training_file
    else
      data_file <- self$test_file
      
    data <- readRDS(file.path(self$processed_folder, data_file))
    self$data <- torch_tensor(data[[1]])
    self$targets <- torch_tensor(data[[2]] + 1L, dtype = torch_long())
  },
  download = function() {
    
    if (self$check_exists())
      return(NULL)
    
    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)
    
    for (r in self$resources) {
      filename <- tail(strsplit(r[1], "/")[[1]], 1)
      destpath <- file.path(self$raw_folder, filename)
      utils::download.file(r[1], destfile = destpath)
      
      if (!tools::md5sum(destpath) == r[2])
        runtime_error("MD5 sums are not identical for file: {r[1}.")
      
    }
    
    inform("Processing...")
    
    training_set <- list(
      read_sn3_pascalvincent(file.path(self$raw_folder, 'train-images-idx3-ubyte.gz')),
      read_sn3_pascalvincent(file.path(self$raw_folder, 'train-labels-idx1-ubyte.gz'))
    )
    
    test_set <- list(
      read_sn3_pascalvincent(file.path(self$raw_folder, 't10k-images-idx3-ubyte.gz')),
      read_sn3_pascalvincent(file.path(self$raw_folder, 't10k-labels-idx1-ubyte.gz'))
    )
    
    saveRDS(training_set, file.path(self$processed_folder, self$training_file))
    saveRDS(test_set, file.path(self$processed_folder, self$test_file))
    
    inform("Done!")
    
  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img <- self$data[index, ,]
    target <- self$targets[index]
    
    if (!is.null(self$transform))
      img <- self$transform(img)
    
    if (!is.null(self$target_transform))
      target <- self$target_transform(target)
    
    list(img, target)
  },
  .length = function() {
    self$data$shape[1]
  },
  active = list(
    raw_folder = function() {
      file.path(self$root, "mnist", "raw")
    },
    processed_folder = function() {
      file.path(self$root, "mnist", "processed")
    }
  )
)

#' Kuzushiji-MNIST
#' 
#' The [Kuzushiji-MNIST dataset](https://github.com/rois-codh/kmnist).
#' 
#' @param root (string): Root directory of dataset where `KMNIST/processed/training.pt`
#'   and  `KMNIST/processed/test.pt` exist.
#' @param train (bool, optional): If TRUE, creates dataset from `training.pt`,
#'   otherwise from `test.pt`.
#' @param download (bool, optional): If true, downloads the dataset from the internet and
#'   puts it in root directory. If dataset is already downloaded, it is not
#'   downloaded again.
#' @param transform (callable, optional): A function/transform that  takes in an PIL image
#'   and returns a transformed version. E.g, `transforms.RandomCrop`
#' @param target_transform (callable, optional): A function/transform that takes in the
#'   target and transforms it.
#'   
#' @export
kmnist_dataset <- dataset(
  name = "kminst_dataset",
  inherit = mnist_dataset,
  resources = list(
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134")
  ),
  classes = c('o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo')
)

read_sn3_pascalvincent <- function(path) {
  x <- gzfile(path, open = "rb")
  on.exit({close(x)})
  
  magic <- readBin(x, endian = "big", what = integer(), n = 1)
  n_dimensions <- magic %% 256
  ty <- magic %/% 256
  
  dim <- readBin(x, what = integer(), size = 4, endian = "big",
          n = n_dimensions)
  
  a <- readBin(
    x, 
    what = "int", endian = "big", n = prod(dim),
    size = 1, signed = FALSE
  )
  
  a <- array(a, dim = rev(dim))
  a <- aperm(a, perm = rev(seq_along(dim)))
  a
}
