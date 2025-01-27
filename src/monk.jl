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

vector_train_files = ["../dataset/monks/monks-1.train", "../dataset/monks/monks-2.train", "../dataset/monks/monks-3.train"]
vector_test_files = ["../dataset/monks/monks-1.test", "../dataset/monks/monks-2.test", "../dataset/monks/monks-3.test"]

X_train, Y_train, X_test, Y_test = load_monks_data(vector_train_files[1], vector_test_files[1]) # il primo task: monks-1

X_train = hcat(X_train...)
X_train = convert(Matrix{Float64}, X_train')

X_test = hcat(X_test...)
X_test = convert(Matrix{Float64}, X_test')

Y_train = convert(Vector{Int}, Y_train)
Y_test = convert(Vector{Int}, Y_test)

weight_init = random_initializer(42)
bias_init = zeros_initializer()
activation_functions = [Activations.tanh(), Activations.identity()]
loss = binary_crossentropy()
regularization = momentum_tikhonov()

# mutable struct NeuralNetwork
#     η::Number
#     α::Number
#     batch_sz::Number
#     epochs::Number
#     num_classes::Number
#     layers::AbstractArray{Tensor, 1}
#     biases::AbstractArray{Tensor, 1}
#     activation_functions::AbstractArray{Activation, 1}
#     loss::Loss
#     regularization::MomentumFunction
#     Δw_old::Vector{Matrix{Float64}}

println("Valori unici in Y_train: ", unique(Y_train))
println("Valori unici in Y_test: ", unique(Y_test))


nn_prototype = NeuralNetwork(
    0.3,
    0.2,
    100,
    3,
    1, # num_classes
    [(6, 2), (2, 1)],
    weight_init,
    bias_init,
    activation_functions,
    loss,
    regularization
)

grid_params = Dict(
    :η => [0.01, 0.05, 0.1],
    :α => [0.001, 0.01, 0.05],
    :batch_sz => [50, 100, 200],
    :epochs => [2, 3]
)

println("\nRunning Grid Search...")
best_grid_params, best_grid_score = grid_search(
    nn_prototype, X_train, Y_train, grid_params; X_val=X_test, Y_val=Y_test
)

println("Best Grid Search Parameters: $best_grid_params")
println("Best Grid Search Accuracy: $best_grid_score")