# can print ops objects at different levels

    Code
      jit_ops
    Output
      Object of class <torch_ops>

---

    Code
      jit_ops$sparse
    Output
      <torch_ops>: Handle to namespace  sparse 

---

    Code
      jit_ops$prim$ChunkSizes
    Output
      function_schema (name = prim::ChunkSizes)
      arguments: <none> 
      returns: <none> 

---

    Code
      jit_ops$aten$fft_fft
    Output
      function_schema (name = aten::fft_fft)
      arguments: self - TensorType, n - OptionalType, dim - IntType, norm - OptionalType 
      returns: TensorType 

