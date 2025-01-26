# Include the dependencies
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

using Random
using Images

function load_mnist_data(base_path::String)
    X, y = [], []
    for digit in 0:9
        folder_path = joinpath(base_path, string(digit))
        for file in readdir(folder_path)
            img_path = joinpath(folder_path, file)
            img = load(img_path)
            img_mat = Float64.(img)
            img_flat = reshape(img_mat, :)
            push!(X, img_flat)
            push!(y, digit)
        end
    end

    # Convert to Matrix form
    X = hcat(X...)'
    n = size(X, 1)

    # Shuffle the data
    perm = shuffle(1:n)
    X = X[perm, :]
    y = y[perm]

    # Split into training and validation sets
    train_size = Int(0.8 * n)
    X_train = X[1:train_size, :]
    Y_train = y[1:train_size]
    X_val = X[train_size+1:end, :]
    Y_val = y[train_size+1:end]

    return X_train, Y_train, X_val, Y_val
end

base_path = "../dataset/mnist/trainingSet/trainingSet"

X_train, Y_train, X_val, Y_val = load_mnist_data(base_path)

weight_init = random_initializer(42)
bias_init = zeros_initializer()
activation_functions = [Activations.tanh(), Activations.identity()]
loss = softmax_crossentropy()
regularization = momentum_tikhonov()

nn_prototype = NeuralNetwork(
    0.3,
    0.2,
    100,
    3,
    10,
    [(784, 128), (128, 10)],
    weight_init,
    bias_init,
    activation_functions,
    loss,
    regularization
)

random_params = Dict(
    :η => [0.01],
    :epochs => [2, 3]
)

println("\nRunning Random Search with Cross-Validation...")
best_random_params, best_random_score = random_search(
    nn_prototype, X_train, Y_train, random_params, 10, cv=true, k=5
)

println("Best Random Search Parameters: $best_random_params")
println("Best Random Search Accuracy: $best_random_score")

# Random Search withOUT Cross-Validation
println("\nRunning Random Search withOUT Cross-Validation...")
best_random_params, best_random_score = random_search(
    nn_prototype, X_train, Y_train, random_params, 10; cv=false, X_val=X_val, Y_val=Y_val
)

println("Best Random Search Parameters: $best_random_params")


# -----

# Grid Search with Cross-Validation
grid_params = Dict(
    :η => [0.01],
    :epochs => [2, 3]
)

println("\nRunning Grid Search with Cross-Validation...")
best_grid_params, best_grid_score = grid_search(
    nn_prototype, X_train, Y_train, grid_params; cv=true, k=5
)
println("Best Grid Search Parameters: $best_grid_params")
println("Best Grid Search Accuracy: $best_grid_score")


println("\nRunning Grid Search withOUT Cross-Validation...")
best_grid_params, best_grid_score = grid_search(
    nn_prototype, X_train, Y_train, grid_params; cv=false, X_val=X_val, Y_val=Y_val
)
println("Best Grid Search Parameters: $best_grid_params")
println("Best Grid Search Accuracy: $best_grid_score")

