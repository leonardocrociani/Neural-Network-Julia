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

loss_fn = mse()
regularization = momentum_tikhonov()

grid_params = Dict(
	:η => [0.1, 0.3, 0.8],
	:α => [0.01, 0.5, 0.1],
	:batch_sz => [50, 100, 200],
	:epochs => [10, 50]
)

keys_order = (:η, :α, :batch_sz, :epochs)
combinations = Iterators.product([grid_params[k] for k in keys_order]...)
scores = Float64[]
params = []
# Iterate over each combination
for combo in combinations
	# Extract the values for each parameter
	η, α, batch_sz, epochs = combo
	# println("η: $η, α: $α, batch_sz: $batch_sz, epochs: $epochs")
	nn_prototype = NeuralNetwork(
			η,
			α,
			batch_sz,
			epochs,
			2, # num_classes
			[(6, 50), (50, 2)],
			weight_init,
			bias_init,
			[Activations.tanh(), Activations.sigmoid()],
			loss_fn,
			momentum_tikhonov()
	)
	train(nn_prototype, X_train, Y_train)
	correct = 0
	total = 0
	for i in axes(Y_test, 1)
			X_in = X_test[i:i,:]
			X_in = Tensor(X_in)
			Y_true = Y_test[i]
			layer = nn_prototype.activation_functions[1](X_in * nn_prototype.layers[1] + nn_prototype.biases[1])
			for i in 2:size(nn_prototype.layers, 1)
				layer = nn_prototype.activation_functions[i](layer * nn_prototype.layers[i] + nn_prototype.biases[i])
			end
			# println("$(layer.data) : $(argmax(layer.data, dims=2)[1][2] - 1) vs $(Int(Y_true))")
			if argmax(layer.data, dims=2)[1][2] - 1 == Int(Y_true)
				correct += 1
			end
			total += 1
	end
	push!(params, combo)
	push!(scores, correct/total)
end
println("$(params[argmax(scores)]): $(maximum(scores))")