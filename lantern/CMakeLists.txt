cmake_minimum_required(VERSION 3.16)

project(lantern)


############################################################
# Torch
############################################################

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_BINARY_DIR}/libtorch/share/cmake/Torch")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



############################################################
# Library
############################################################

set(LANTERN_SRC
    src/lantern.cpp
    src/TensorOptions.cpp
    src/Dtype.cpp
    src/Tensor.cpp
    src/Device.cpp
    src/utils.cpp
    src/MemoryFormat.cpp
    src/Generator.cpp
    src/QScheme.cpp
    src/TensorList.cpp
    src/Scalar.cpp
    src/Dimname.cpp
    src/Delete.cpp
    src/Reduction.cpp
    src/Quantization.cpp
    src/Autograd.cpp
    src/Function.cpp
    src/Layout.cpp
    src/Indexing.cpp
    src/Cuda.cpp
    src/NNUtilsRnn.cpp
    src/Storage.cpp
    src/Save.cpp
    src/Contrib/Sparsemax.cpp
    src/Threads.cpp
    src/Trace.cpp
    src/Stack.cpp
    src/Allocator.cpp
    src/Backends.cpp
    src/JITTypes.cpp
    src/ScriptModule.cpp
    src/IValue.cpp
    src/Compile.cpp
)

set(LANTERN_SRC
      ${LANTERN_SRC}
      src/Contrib/SortVertices/sort_vert_cpu.cpp
  )
  add_library(lantern SHARED ${LANTERN_SRC})


#target_compile_definitions(lantern PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

add_library(lantern::library ALIAS lantern)

target_include_directories(lantern PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(lantern ${TORCH_LIBRARIES})
set_property(TARGET lantern PROPERTY CXX_STANDARD 17)

############################################################
# Tests
############################################################

add_executable(lanterntest
    tests/init.cpp
    tests/main.cpp
)

target_include_directories(lanterntest PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    tests/
)

target_link_libraries(lanterntest ${CMAKE_DL_LIBS})
