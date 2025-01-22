# Ensure your module is included correctly
include("lib/Tensor.jl")

using .Tensors

x = [0.0, 42.0, 5.0]
println(size(x)) # expected output: (3,)

x = Tensor(x)
println(size(x.data)) # expected output: (1, 3)