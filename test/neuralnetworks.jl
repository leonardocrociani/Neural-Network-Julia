include("../src/lib/Tensor.jl")
include("../src/lib/Initializer.jl")
include("../src/lib/Loss.jl")
include("../src/lib/Activation.jl")
include("../src/lib/Regularization.jl")
include("../src/lib/NeuralNetwork.jl")

using Test
using Random
using Images


using .Tensors
using .Initializers
using .Losses
using .Activations
using .Regularizations
using .NeuralNetworks

weight_init = random_initializer(42)
bias_init = zeros_initializer()
loss = softmax_crossentropy()
regularization = momentum()
activation_functions = [ tanh(), Activations.identity() ]

nn = NeuralNetwork(
	0.8,
	0.5,
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


# reading images 
base_path = "./dataset/mnist/trainingSet/trainingSet"

X = [] # image pixel data
y = [] # digit label

for digit in 0:9 # iterating over images and computing the matrix and flatting it
    folder_path = joinpath(base_path, string(digit))
    for file in readdir(folder_path)
        img_path = joinpath(folder_path, file)
        img = load(img_path)
        img_mat = Float64.(img)
        img_flat = reshape(img_mat,:)
        push!(X, img_flat)
        push!(y, digit)
    end
end
# X will look like (784,N)
X = hcat(X...)' # trasposing to (N,784)

n = size(X, 1) 
# we shuffle the data in the same random order
perm = shuffle(1:n)
X = X[perm, :]
y = y[perm];

## now lets split up data in training and testing datasets ( provided testing set was broken)

train_size = Int(0.8 * size(X,1))

X_train = X[1:train_size, :]
Y_train = y[1:train_size]

X_test = X[train_size+1:end, :]
Y_test = y[train_size+1:end]


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

	if argmax(layer.data, dims=2)[1][2] - 1 == Y_true # -1 because digits start at 0
		global correct +=1
	end
	global total += 1

end

println("accuracy: $(correct/total)")

