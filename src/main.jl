# Ensure your module is included correctly
include("lib/Tensor.jl")

using .Tensors

x = [0.0, 42.0, 5.0]
println(size(x))

x = Tensor(x)
println(size(x.data))