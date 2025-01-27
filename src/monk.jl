include("../src/lib/DataLoader.jl")
include("../src/lib/Tensor.jl")
include("../src/lib/Regularization.jl")
include("../src/lib/Initializer.jl")
include("../src/lib/Loss.jl")
include("../src/lib/Activation.jl")
include("../src/lib/NeuralNetwork.jl")
include("../src/lib/HyperparamSearch.jl")

using .Regularizations
using .Tensors
using .Initializers
using .Losses
using .Activations
using .NeuralNetworks
using .HyperparamSearch
using .DataLoader

vector_train_files = ["./dataset/monks/monks-1.train", "./dataset/monks/monks-2.train", "./dataset/monks/monks-3.train"]
vector_test_files = ["./dataset/monks/monks-1.test", "./dataset/monks/monks-2.test", "./dataset/monks/monks-3.test"]

X_train, Y_train, X_test, Y_test = load_monks_data(vector_train_files[1], vector_test_files[1]) # il primo task: monks-1

X_train = hcat(X_train...)
X_train = convert(Matrix{Float64}, X_train')

X_test = hcat(X_test...)
X_test = convert(Matrix{Float64}, X_test')

Y_train = convert(Vector{Float64}, Y_train)
Y_test = convert(Vector{Float64}, Y_test)

weight_init = random_initializer(42)
bias_init = zeros_initializer()

loss = mse()
regularization = momentum_tikhonov()
# Training the neural network with adjusted parameters
#nn = NeuralNetwork(
#    0.01,            # Increased learning rate
#    0.9,             # Increased momentum (α)
#    32,              # Larger batch size
#    200,             # Increased epochs for more learning
#    1,               # num_classes
#    [(6, 4), (4, 1)], # More complex network architecture
#    weight_init,
#    bias_init,
#    [Activations.tanh(), Activations.sigmoid()], # Use sigmoid for output layer
#    cross_entropy(), # Changed loss to cross-entropy for binary classification
#    regularization
#)

# grid_params = Dict(
# 	:η => [0.01, 0.05, 0.1],
# 	:α => [0.9, 0.01, 0.05],
# 	:batch_sz => [32, 100, 10],
# 	:epochs => [50, 200]
# )

# keys_order = (:η, :α, :batch_sz, :epochs)
# combinations = Iterators.product([grid_params[k] for k in keys_order]...)

# # Iterate over each combination
# for combo in combinations
# 	# Extract the values for each parameter
# 	η, α, batch_sz, epochs = combo

# 	println("η: $η, α: $α, batch_sz: $batch_sz, epochs: $epochs")

# 		nn_prototype = NeuralNetwork(
#             η,
#             α,
#             batch_sz,
#             epochs,
#             1, # num_classes
#             [(6, 100), (100, 1)],
#             weight_init,
#             bias_init,
#             [Activations.tanh(), Activations.sigmoid()],
#             loss,
#             regularization
# 		)

# 		train(nn_prototype, X_train, Y_train)

# 		correct = 0
# 		total = 0
# 		for i in axes(Y_test, 1)
# 				X_in = X_test[i:i,:]
# 				X_in = Tensor(X_in)
# 				Y_true = Y_test[i]

# 				layer = nn_prototype.activation_functions[1](X_in * nn_prototype.layers[1] + nn_prototype.biases[1])
# 				for i in 2:size(nn_prototype.layers, 1)
# 					layer = nn_prototype.activation_functions[i](layer * nn_prototype.layers[i] + nn_prototype.biases[i])
# 				end

#                 output = layer.data[1]

#                 predizione = output > 0.7 ? 1 : 0

# 				if predizione == Y_true
# 						correct +=1
# 				end
# 				total += 1

# 		end

# 		println("accuracy: $(correct/total)") 
# end


nn_prototype = NeuralNetwork(
    0.1,
    0.5,
    32,
    50,
    1, # num_classes
    [(6, 100), (100, 100), (100, 1)],
    weight_init,
    bias_init,
    [Activations.tanh(), Activations.tanh(), Activations.sigmoid()],
    loss,
    regularization
)

train(nn_prototype, X_train, Y_train, verbose=true)

correct = 0
total = 0
for i in axes(Y_test, 1)
    X_in = X_test[i:i, :]
    X_in = Tensor(X_in)
    Y_true = Y_test[i]

    layer = nn_prototype.activation_functions[1](X_in * nn_prototype.layers[1] + nn_prototype.biases[1])
    for i in 2:size(nn_prototype.layers, 1)
        layer = nn_prototype.activation_functions[i](layer * nn_prototype.layers[i] + nn_prototype.biases[i])
    end

    output = layer.data[1]

    predizione = output > 0.5 ? 1 : 0

    if predizione == Y_true
        global correct += 1
    end
    global total += 1

end

println("accuracy: $(correct/total)")

