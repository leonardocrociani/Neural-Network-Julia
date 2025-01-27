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

loss = mse()
regularization = momentum_tikhonov()
# Training the neural network with adjusted parameters
nn = NeuralNetwork(
    0.01,            # Increased learning rate
    0.9,             # Increased momentum (α)
    32,              # Larger batch size
    200,             # Increased epochs for more learning
    1,               # num_classes
    [(6, 4), (4, 1)], # More complex network architecture
    weight_init,
    bias_init,
    [Activations.tanh(), Activations.sigmoid()], # Use sigmoid for output layer
    cross_entropy(), # Changed loss to cross-entropy for binary classification
    regularization
)

# Training the neural network
train(nn, X_train, Y_train, verbose=true)

correct = 0
total = 0
for i in eachindex(Y_test)
    X_in = X_test[i:i,:]
    X_in = Tensor(X_in)
    Y_true = Y_test[i]
    layer = nn.activation_functions[1](X_in * nn.layers[1] + nn.biases[1])
    for i in 2:size(nn.layers, 1)
        layer = nn.activation_functions[i](layer * nn.layers[i] + nn.biases[i])
    end
    prediction = (layer.data[1] >= 0.5) ? 1 : 0
    println("output ", layer.data[1], " ", prediction, " ", Y_true)
    if prediction == Y_true # -1 because digits start at 0
        global correct +=1
    end
    global total += 1
end

println("accuracy: $(correct/total)")
