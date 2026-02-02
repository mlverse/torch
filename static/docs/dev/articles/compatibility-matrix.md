# CUDA compatibility matrix

Finding the right combination of libtorch, {torch}, and CUDA can be
tricky. This guide provides a clear compatibility matrix to help you
set-up your environment.

## {torch} versions compatibility

| {torch} | libtorch | Suitable CUDA versions  |
|:-------:|:--------:|:-----------------------:|
| 0.16.x  |  2.7.1   |       12.6, 12.8        |
| 0.15.x  |  2.5.1   |       11.8, 12.4        |
| 0.14.x  |  2.5.1   |       11.8, 12.4        |
| 0.13.0  |  2.0.1   |       11.7, 11.8        |
| 0.12.0  |  2.0.1   |       11.7, 11.8        |
| 0.11.0  |  1.13.1  |       11.6, 11.7        |
| 0.10.0  |  1.13.1  | 11.6 (Linux only), 11.7 |

torch compatibility matrix

## CUDA version compatibility

| CUDA version |  Suitable {torch} version   |
|:------------:|:---------------------------:|
|     11.6     | 0.10.0 (Linux only), 0.11.0 |
|     11.7     |      0.10.0 to 0.13.0       |
|     11.8     |      0.12.0 to 0.15.x       |
|     12.4     |      0.14.x to 0.15.x       |
|     12.6     |           0.16.x            |
|     12.8     |           0.16.x            |
