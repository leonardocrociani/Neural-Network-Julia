# Ensure your module is included correctly
include("./lib/DataTypes.jl")

using .DataTypes


x = Value(4)
y = Value(4)

println(x, " => data: $(x.data), grad: $(x.grad), op: $(x.op)")
println(y)

println(x == y) # should be false

z = x # reference
println(x == z) # should be true

a = x + y
println(a, " => data: $(a.data), grad: $(a.grad), op: $(a.op)") # Should show addition operation

# backpropagation testing

x = Value(2)
y = Value(3)

a = Value(1)
b = Value(4)

w = x + y
z = a + b

c = w + z

backward(c)

println(x.grad, y.grad, a.grad, b.grad) # should be 1.0 1.0 1.0 1.0
