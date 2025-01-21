include("lib/DataTypes.jl") # modules are uppercased :|
using .DataTypes

x = Value(4)
y = Value(4)

println(x, " => data: $(x.data), grad: $(x.grad), op: $(x.op)")
println(y)

println(x == y) # should be false

z = x # reference
println(x == z) # should be true

a = x+y
println(a, " => data: $(a.data), grad: $(a.grad), op: $(a.op)")