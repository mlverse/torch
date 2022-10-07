# can print the graph

    graph(%self : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %3 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
      %bias : Tensor = prim::GetAttr[name="bias"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%self)
      %4 : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu) = aten::linear(%3, %weight, %bias)
      return (%4)

---

    graph(%self : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %3 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
      %bias : Tensor = prim::GetAttr[name="bias"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%self)
      %4 : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu) = aten::linear(%3, %weight, %bias)
      return (%4)

# graph_for

    graph(%self : __torch__.nn_linear_neebjyloatcfjjfottzlywfy,
          %1 : Tensor):
      %bias : Tensor = prim::GetAttr[name="bias"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%self)
      %5 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu), seen_none=0](%1)
      %6 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu), seen_none=0](%weight)
      %7 : Tensor = prim::profile[profiled_type=Float(10, strides=[1], requires_grad=1, device=cpu), seen_none=0](%bias)
      %4 : Tensor = aten::linear(%5, %6, %7)
      %8 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu), seen_none=0](%4)
       = prim::profile()
      return (%8)

---

    graph(%self : __torch__.nn_linear_neebjyloatcfjjfottzlywfy,
          %1 : Tensor):
      %bias : Tensor = prim::GetAttr[name="bias"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%self)
      %22 : Tensor = aten::linear(%1, %weight, %bias)
      return (%22)

