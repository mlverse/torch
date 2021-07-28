# can print the graph

    graph(%self : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %3 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
      %9 : Tensor = prim::GetAttr[name="bias"](%self)
      %8 : Tensor = prim::GetAttr[name="weight"](%self)
      %4 : Float(10, 10, strides=[1, 10], requires_grad=1, device=cpu) = aten::t(%8)
      %5 : int = prim::Constant[value=1]()
      %6 : int = prim::Constant[value=1]()
      %7 : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu) = aten::addmm(%9, %3, %4, %5, %6)
      return (%7)

---

    graph(%self : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %3 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
      %9 : Tensor = prim::GetAttr[name="bias"](%self)
      %8 : Tensor = prim::GetAttr[name="weight"](%self)
      %4 : Float(10, 10, strides=[1, 10], requires_grad=1, device=cpu) = aten::t(%8)
      %5 : int = prim::Constant[value=1]()
      %6 : int = prim::Constant[value=1]()
      %7 : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu) = aten::addmm(%9, %3, %4, %5, %6)
      return (%7)

