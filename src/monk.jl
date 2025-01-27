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

X_train, Y_train, X_test, Y_test = load_monks_data(vector_train_files[1], vector_test_files[1]) # il primo task: monks-1

X_train = hcat(X_train...)
X_train = convert(Matrix{Float64}, X_train')

X_test = hcat(X_test...)
X_test = convert(Matrix{Float64}, X_test')

Y_train = convert(Vector{Float64}, Y_train)
Y_test = convert(Vector{Float64}, Y_test)

weight_init = random_initializer(42)
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
	[0.1, 0.3, 0.8],
	[0.01, 0.5, 0.1],
	[50, 100, 200],
	[10, 50],
	[(6, 50), (50, 2)],
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