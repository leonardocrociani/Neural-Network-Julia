# Ensure your module is included correctly
include("lib/Tensor.jl")
using Pkg

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
println(weights_layer1.grad)


## Solving MNIST


Pkg.add("Images")
using Images
img_path = "../../dataset/trainingSet/trainingSet/0/img_1.jpg"
img = load(img_path)
img_mat = Float64.(img)
img_flattened = reshape(img_mat,:)
println(size(img_flattened))
# we need to this for each image

# reading images 
base_path = "../../dataset/trainingSet/trainingSet"

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

println(size(X))
println(size(y))

using Random
n = size(X, 1) 
# we shuffle the data in the same random order
perm = shuffle(1:n)
X = X[perm, :]
y = y[perm];

## now lets split up data in training and testing datasets ( provided testing set was broken)

train_size = Int(0.8 * size(X,1))

X_train = X[1:train_size, :]
y_train = y[1:train_size]

X_test = X[train_size+1:end, :]
y_test = y[train_size+1:end]

println(size(X_train))
println(size(y_train))

#initializing the weights and biases
# layer 1
weights1 = Tensor(0.01 * rand(784, 128)) # taking 784 as input and 128 neurons in the layerù
biases1 = Tensor(zeros(128)) # bias for each neuron 
# layer 2
weights2 = Tensor(0.01 * rand(128,10)) # taking 128 inputs ( from 128 neurons in the first layer),  has 10 newrons in the second layer
biases2 = Tensor(zeros(10))
# hyperparameters start
lr = 0.1;
batch_size = 100;
num_classes = 10;
epochs = 3;
# hyperparameters end

global run = 1
for epoch in 1:epochs
    for i in 1:batch_size:size(X_train,1)

        # size of input matrix = (batch_size, 784)
        batch_X = Tensor(X_train[i:i+batch_size-1, :]) # taking first 10 samples
        batch_y = y_train[i:i+batch_size-1] # do not cast this to Tensor!!


        # one hot encoding conversion of batch_y

        batch_y_one_hot = zeros(batch_size, num_classes)
        for batch_ind in 1:batch_size
            batch_y_one_hot[batch_ind,Int.(batch_y)[batch_ind]+1] = 1
        end

        # zero grads
        weights1.grad .= 0
        weights2.grad .= 0
        biases1.grad .= 0
        biases2.grad .= 0

        # layer 1 forward pass
        layer1_out = relu(batch_X * weights1 + biases1);

        # layer 2 forward pass
        layer2_out = layer1_out * weights2 + biases2

        loss = softmax_crossentropy(layer2_out, batch_y_one_hot)

        backward(loss) ## full backward pass for the loss ( small change to every parametrs how does the loss change)

        # updating the weights  and biases according to the gradient scaling the size of the update by the learning rate

        weights1.data = weights1.data - weights1.grad .* lr
        biases1.data = biases1.data - biases1.grad .* lr
        weights2.data = weights2.data - weights2.grad .* lr
        biases2.data = biases2.data - biases2.grad .* lr;
        
        
        if run % 10== 0
            println("Epoch: run, loss: $(round(loss.data[1], digits=3))")
        end
        
        global run += 1
    end
end