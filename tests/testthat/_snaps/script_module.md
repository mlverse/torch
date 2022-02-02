# can print the graph

    graph(%self : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %3 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
      %bias : Tensor = prim::GetAttr[name="bias"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%self)
      %4 : Float(10, 10, strides=[1, 10], requires_grad=1, device=cpu) = aten::t(%weight)
      %5 : int = prim::Constant[value=1]()
      %6 : int = prim::Constant[value=1]()
      %7 : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu) = aten::addmm(%bias, %3, %4, %5, %6)
      return (%7)

---

    graph(%self : __torch__.nn_linear_ydgabwknrsauujvnjgioueiy,
          %3 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
      %bias : Tensor = prim::GetAttr[name="bias"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%self)
      %4 : Float(10, 10, strides=[1, 10], requires_grad=1, device=cpu) = aten::t(%weight)
      %5 : int = prim::Constant[value=1]()
      %6 : int = prim::Constant[value=1]()
      %7 : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu) = aten::addmm(%bias, %3, %4, %5, %6)
      return (%7)

# graph_for

    graph(%self : __torch__.nn_linear_neebjyloatcfjjfottzlywfy,
          %1 : Tensor):
      %2 : int = prim::Constant[value=1]()
      %bias : Tensor = prim::GetAttr[name="bias"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%self)
      %8 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu)](%weight)
      %5 : Tensor = aten::t(%8)
      %9 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)](%1)
      %10 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[1, 10], requires_grad=1, device=cpu)](%5)
      %6 : Tensor = aten::mm(%9, %10) # <string>:3:24
      %11 : Tensor = prim::profile[profiled_type=Float(10, strides=[1], requires_grad=1, device=cpu)](%bias)
      %12 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu)](%6)
      %7 : Tensor = aten::add(%11, %12, %2) # <string>:3:17
      %13 : Tensor = prim::profile[profiled_type=Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu)](%7)
       = prim::profile()
      return (%13)

---

    graph(%self : __torch__.nn_linear_neebjyloatcfjjfottzlywfy,
          %1 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
      %bias : Float(10, strides=[1], requires_grad=1, device=cpu) = prim::GetAttr[name="bias"](%self)
      %weight : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu) = prim::GetAttr[name="weight"](%self)
      %32 : Float(10, strides=[1], requires_grad=1, device=cpu), %33 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu), %34 : Float(10, 10, strides=[10, 1], requires_grad=1, device=cpu), %35 : bool = prim::RequiresGradCheck[types=[Tensor(requires_grad=1), Tensor(requires_grad=0), Tensor(requires_grad=1)]](%bias, %1, %weight)
      %36 : Tensor = prim::If(%35)
        block0():
          %15 : Tensor = prim::DifferentiableGraph_0(%32, %33, %34)
          -> (%15)
        block1():
          %48 : Function = prim::Constant[name="fallback_function", fallback=1]()
          %49 : (Tensor) = prim::CallFunction(%48, %bias, %1, %weight)
          %50 : Tensor = prim::TupleUnpack(%49)
          -> (%50)
      return (%36)
    with prim::DifferentiableGraph_0 = graph(%bias : Tensor(requires_grad=0),
          %15 : Tensor(requires_grad=0),
          %weight : Tensor(requires_grad=0)):
      %18 : Tensor(requires_grad=0) = aten::t(%weight)
      %12 : Tensor(requires_grad=0) = aten::mm(%15, %18) # <string>:3:24
      %4 : int = prim::Constant[value=1]()
      %5 : Tensor(requires_grad=0) = aten::add(%bias, %12, %4) # <string>:3:17
      %68 : int[] = aten::size(%bias) # <string>:3:44
      %69 : int[] = aten::size(%5) # <string>:3:55
      %70 : int[]? = aten::_size_if_not_equal(%68, %69) # <string>:3:19
      %71 : int[] = aten::size(%12) # <string>:3:93
      %73 : int[]? = aten::_size_if_not_equal(%71, %69) # <string>:3:68
      %74 : (int[]?, int[]?) = prim::TupleConstruct(%70, %73)
      %self_size.4 : int[]?, %other_size.4 : int[]? = prim::TupleUnpack(%74)
      return (%5, %15, %18, %self_size.4, %other_size.4)

