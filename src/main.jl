# Ensure your module is included correctly
include("lib/Tensor.jl")

using .Tensors

x = [0.0, 42.0, 5.0]
println(size(x)) # expected output: (3,)

x = Tensor(x)
println(size(x.data)) # expected output: (1, 3)

x = Tensor([1.0 2.0 ; 3.0 4.0])

println(size(x))

x[2,2] = 5.0

println(x)  
println(x[2,2,]) # expected output: 5.0

a = Tensor([1.2, -5.4, 10.3, -2.0])
b = relu(a)

println(b)

backward(b)

println(a.grad)
println(b.grad)


# softmax_crossentropy test
output_layer = Tensor(rand(3, 3)) # Matrix 3x3 with 3 inputs and 3 outputs
y_true = [0 0 1; 0 1 0; 1 0 1]
println(softmax_crossentropy(output_layer, y_true))