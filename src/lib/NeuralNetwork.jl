module NeuralNetworks

	using ..Tensors
	using ..Initializers
	using ..Losses
	using ..Activations
	using ..Regularizations

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
		regularization::MomentumFunction
		Δw_old::Vector{Matrix{Float64}}

		function NeuralNetwork(η::Number , α::Number,
				batch_sz::Number, epochs::Number, num_classes::Number,
				layer_sizes::AbstractArray{<:Tuple{<:Number, <:Number}},
				initialize_weights_function::Initializer, initialize_biases_function::Initializer,
				activation_functions::AbstractArray{Activation, 1}, loss::Loss,
				regularization_function::MomentumFunction)
			@assert size(activation_functions, 1) === size(layer_sizes, 1)
			layers = Vector{Tensor}(undef, size(layer_sizes, 1))
			old_layers = Vector{Matrix{Float64}}(undef, size(layer_sizes, 1))
			biases = Vector{Tensor}(undef, size(layer_sizes, 1))
			for index in eachindex(layer_sizes)
				sz = layer_sizes[index]
				layers[index] = Tensor(0.01 * initialize_weights_function(sz[1], sz[2]))
				biases[index] = Tensor(0.01 * initialize_biases_function(1, sz[2]))
				old_layers[index] = zeros(sz[1], sz[2])
			end
			new(η, α, batch_sz, epochs, num_classes, layers, biases, activation_functions, loss, regularization_function, old_layers)
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
			# Adjusted the batch range to avoid out-of-bounds errors
			batch_end = min(i + nn.batch_sz - 1, size(X_train, 1))
			X_batch = Tensor(X_train[i:batch_end, :])
			Y_batch = Y_train[i:batch_end]

			# Adjust the size of Y_batch_encoded dynamically for the last batch
			batch_size_actual = size(X_batch, 1)
			if nn.num_classes == 1
				# Per binary crossentropy
				Y_batch_encoded = reshape(Y_batch, :, 1)
			else
				# Per softmax crossentropy
				Y_batch_encoded = zeros(batch_size_actual, nn.num_classes)
				for batch_index in 1:batch_size_actual
					class_index = Int(Y_batch[batch_index]) + 1
					if class_index > nn.num_classes || class_index < 1
						error("Classe non valida: $(Y_batch[batch_index]) per num_classes=$(nn.num_classes)")
					end
					Y_batch_encoded[batch_index, class_index] = 1
				end
			end


			# Reset gradients
			for layer in nn.layers
				layer.grad .= 0
			end
			for bias in nn.biases
				bias.grad .= 0
			end

			# Forward pass
			layer = nn.activation_functions[1](X_batch * nn.layers[1] + nn.biases[1])
			for j in 2:size(nn.layers, 1)
				layer = nn.activation_functions[j](layer * nn.layers[j] + nn.biases[j])
			end

			# Compute loss and perform backpropagation
			loss = nn.loss(layer, Y_batch_encoded)
			nn.regularization(loss, nn.layers, nn.Δw_old, nn.η, nn.α)

			# Print verbose output
			if verbose && run % 10 == 0
				println("[$(epoch)/$(nn.epochs)] Loss: $(loss.data[1])")
			end
			run += 1
		end
		end
	end

end