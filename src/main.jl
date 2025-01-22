# Ensure your module is included correctly
include("lib/DataTypes.jl")

using .DataTypes

m = Value(2)
x = Value(3)
q = Value(7)
s = Value(2)

s = inv(s) # 1/s

r = Value(10)

y = m / r * x - q + s

println(y * 4)

backward(y)

println(m.grad) 

println(q.grad)

y = Value(2)

println(-y)