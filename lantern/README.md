# Lantern

Lantern provides a C API to libtorch. To build in OS X run:

```
pip install cmake
mkdir build
cd build
cmake ..
make
```

## Headers

Re-generating the headers requires `declarations.yaml` from building `libtorch` from source or downloading a pre-built version from [dfalbel/declarations](https://github.com/dfalbel/declarations/releases/tag/declarations).

To re-generate, first build `lanterngen`:

```
cd headers
mkdir build
cd build
cmake ..
make
```

Followed by running `lanterngen` with the downloaded declarations file:

```
./lanterngen ~/Downloads/declarations-v1.5 ../../src/lantern.cpp ../../include/lantern/lantern.h
```
