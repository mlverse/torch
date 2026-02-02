# Number of threads

Get and set the numbers used by torch computations.

## Usage

``` r
torch_set_num_threads(num_threads)

torch_set_num_interop_threads(num_threads)

torch_get_num_interop_threads()

torch_get_num_threads()
```

## Arguments

- num_threads:

  number of threads to set.

## Details

For details see the [CPU threading
article](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html?highlight=set_num_threads)
in the PyTorch documentation.

## Note

torch_set_threads do not work on macOS system as it must be 1.
