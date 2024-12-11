# can print the graph

    graph(%self.1 : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %x.1 : Tensor):
      %training : bool = prim::GetAttr[name="training"](%self.1)
      %19 : Tensor = prim::If(%training) # <string>:3:6
        block0():
          %7 : Tensor = prim::CallMethod[name="trainforward"](%self.1, %x.1) # <string>:4:15
          -> (%7)
        block1():
          %10 : Tensor = prim::CallMethod[name="evalforward"](%self.1, %x.1) # <string>:6:15
          -> (%10)
      return (%19)

---

    graph(%self.1 : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %x.1 : Tensor):
      %training : bool = prim::GetAttr[name="training"](%self.1)
      %19 : Tensor = prim::If(%training) # <string>:3:6
        block0():
          %7 : Tensor = prim::CallMethod[name="trainforward"](%self.1, %x.1) # <string>:4:15
          -> (%7)
        block1():
          %10 : Tensor = prim::CallMethod[name="evalforward"](%self.1, %x.1) # <string>:6:15
          -> (%10)
      return (%19)

# graph_for

    graph(%self.1 : __torch__.nn_linear_neebjyloatcfjjfottzlywfy,
          %x.1 : Tensor):
      %training : bool = prim::GetAttr[name="training"](%self.1)
      %3 : Tensor = prim::If(%training) # <string>:3:6
        block0():
          %bias.1 : Tensor = prim::GetAttr[name="bias"](%self.1)
          %weight.1 : Tensor = prim::GetAttr[name="weight"](%self.1)
          %10 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu), seen_none=0](%x.1)
          %11 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu), seen_none=0](%weight.1)
          %12 : Tensor = prim::profile[profiled_type=Float(10, strides=[1], requires_grad=1, device=cpu), seen_none=0](%bias.1)
          %6 : Tensor = aten::linear(%10, %11, %12)
          %13 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu), seen_none=0](%6)
          -> (%13)
        block1():
          %bias : Tensor = prim::GetAttr[name="bias"](%self.1)
          %weight : Tensor = prim::GetAttr[name="weight"](%self.1)
          %14 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%x.1)
          %15 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%weight)
          %16 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%bias)
          %9 : Tensor = aten::linear(%14, %15, %16)
          %17 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%9)
          -> (%17)
      %18 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu), seen_none=0](%3)
       = prim::profile()
      return (%18)

---

    graph(%self.1 : __torch__.nn_linear_neebjyloatcfjjfottzlywfy,
          %x.1 : Tensor):
      %training : bool = prim::GetAttr[name="training"](%self.1)
      %3 : Tensor = prim::If(%training) # <string>:3:6
        block0():
          %bias.1 : Tensor = prim::GetAttr[name="bias"](%self.1)
          %weight.1 : Tensor = prim::GetAttr[name="weight"](%self.1)
          %42 : Tensor = aten::linear(%x.1, %weight.1, %bias.1)
          -> (%42)
        block1():
          %bias : Tensor = prim::GetAttr[name="bias"](%self.1)
          %weight : Tensor = prim::GetAttr[name="weight"](%self.1)
          %47 : Tensor = aten::linear(%x.1, %weight, %bias)
          -> (%47)
      return (%3)

