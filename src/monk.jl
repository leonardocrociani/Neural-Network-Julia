include("../src/lib/DataLoader.jl")
include("../src/lib/Tensor.jl")
include("../src/lib/Regularization.jl")
include("../src/lib/Initializer.jl")
include("../src/lib/Loss.jl")
include("../src/lib/Score.jl")
include("../src/lib/Activation.jl")
include("../src/lib/NeuralNetwork.jl")

using .Regularizations
using .Tensors
using .Initializers
using .Losses
using .Activations
using .NeuralNetworks
using .DataLoader
using .Scores

vector_train_files = ["./dataset/monks/monks-1.train", "./dataset/monks/monks-2.train", "./dataset/monks/monks-3.train"]
vector_test_files = ["./dataset/monks/monks-1.test", "./dataset/monks/monks-2.test", "./dataset/monks/monks-3.test"]

train_dataset = load_monks_data(vector_train_files[1])
test_dataset = load_monks_data(vector_test_files[1])


X_train = convert(Matrix{Int64}, reduce(hcat, [Float64.(row) for row in [data[2] for data in train_dataset]])')
Y_train = [data[3] for data in train_dataset]


X_test = convert(Matrix{Int64}, reduce(hcat, [Float64.(row) for row in [data[2] for data in test_dataset]])')
Y_test = [data[3] for data in test_dataset]

weight_init = random_initializer()
bias_init = zeros_initializer()

activation_fns = [Activations.sigmoid(), Activations.sigmoid()]
loss_fn = mse()
regularization = momentum_tikhonov()

grid_params = Dict(
	:η => [0.1, 0.3, 0.8],
	:α => [0.01, 0.5, 0.1],
	:batch_sz => [50, 100, 200],
	:epochs => [10, 50]
)

grid_search(
	[0.875, 0.85, 0.8],
	[0.875, 0.85, 0.8],
	[40, 10, 20],
	[400, 200],
	[(17, 50), (50, 2)],
	weight_init,
	bias_init,
	activation_fns,
	loss_fn,
	regularization,
	X_train,
	Y_train,
	X_test,
	Y_test,
	monk(),
	verbose=true
)