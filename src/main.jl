# Ensure your module is included correctly
include("lib/Tensor.jl")

using .Tensors

inputs = Tensor(rand(2, 3))
y_true = [0 0 1 0 0; 0 1 0 0 0]

weights_layer1 = Tensor(rand(3, 4))
bias_layer1 = Tensor(ones(1, 4))

weights_layer2 = Tensor(rand(4, 5))
bias_layer2 = Tensor(ones(1, 5))

layer1_output = relu(inputs * weights_layer1 + bias_layer1)
layer2_output = layer1_output * weights_layer2 + bias_layer2

loss = softmax_crossentropy(layer2_output, y_true, grad=true)
println(loss)
# println(layer2_output.grad)
backward(loss)
# println(layer2_output.grad)

println(weights_layer1)
println()
println(weights_layer1.grad)