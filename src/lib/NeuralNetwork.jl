module NeuralNetworks

	using ..Tensors
	using ..Initializers
	using ..Losses
	using ..Activations

	export NeuralNetwork
	export train

	mutable struct NeuralNetwork
		η::Number # learning rate
		α::Number # momentum
		batch_sz::Number
		epochs::Number
		num_classes::Number
		layers::AbstractArray{Tensor, 1}
		biases::AbstractArray{Tensor, 1}
		activation_functions::AbstractArray{Activation, 1}
		loss::Loss

		function NeuralNetwork(η::Number , α::Number,
				batch_sz::Number, epochs::Number, num_classes::Number,
				layer_sizes::AbstractArray{<:Tuple{<:Number, <:Number}},
				initialize_weights_function::Initializer, initialize_biases_function::Initializer,
				activation_functions::AbstractArray{Activation, 1}, loss::Loss)
			@assert size(activation_functions, 1) === size(layer_sizes, 1)
			layers = Vector{Tensor}(undef, size(layer_sizes, 1))
			biases = Vector{Tensor}(undef, size(layer_sizes, 1))
			for index in eachindex(layer_sizes)
				sz = layer_sizes[index]
				layers[index] = Tensor(0.01 * initialize_weights_function(sz[1], sz[2]))
				biases[index] = Tensor(0.01 * initialize_biases_function(1, sz[2]))
			end
			new(η, α, batch_sz, epochs, num_classes, layers, biases, activation_functions, loss)
		end
	end

	# Pretty printing
	import Base.show
	function Base.show(io::IO, nn::NeuralNetwork)
		print(io,
			"η: $(nn.η) α: $(nn.α) batch size: $(nn.batch_sz) epochs: $(nn.epochs) num_classes: $(nn.num_classes)\nlayers: $(nn.layers)\nbiases: $(nn.biases)")
	end

	function train(nn::NeuralNetwork, X_train::Matrix{<:Number}, Y_train::AbstractArray{<:Any};
				verbose::Bool=false)
		run = 0
		for epoch in 1:nn.epochs
			for i in 1:nn.batch_sz:size(X_train, 1)
				X_batch = Tensor(X_train[i:i+nn.batch_sz-1, :])
				Y_batch = Y_train[i:i+nn.batch_sz-1]

				Y_batch_encoded = zeros(nn.batch_sz, nn.num_classes)
				for batch_index in eachindex(1:nn.batch_sz)
					Y_batch_encoded[batch_index, Int.(Y_batch)[batch_index] + 1] = 1
				end

				for layer in nn.layers
					layer.grad .= 0
				end
				for bias in nn.biases
					bias.grad .= 0
				end

				layer = nn.activation_functions[1](X_batch * nn.layers[1] + nn.biases[1])
				for j in 2:size(nn.layers, 1)
					layer = nn.activation_functions[j](layer * nn.layers[j] + nn.biases[j])
				end
				loss = nn.loss(layer, Y_batch_encoded)

				### TODO: This block should be inside the Regularization struct
				Tensors.backward(loss)
				for layer in nn.layers
					layer.data = layer.data - layer.grad .* nn.η
				end
				###

				if verbose && run % 10 == 0
					println("[$(epoch)/$(nn.epochs)] Loss: $(loss.data[1])")
				end
				run += 1

			end
		end
	end
end