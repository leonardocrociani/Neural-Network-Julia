module NeuralNetworks

	using Random
	using Statistics

	using ..Tensors
	using ..Initializers
	using ..Losses
	using ..Activations
	using ..Regularizations
	using ..Scores

	export NeuralNetwork
	export train
	export grid_search

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
		err::Float64 = Inf
		for epoch in 1:nn.epochs
			for i in 1:nn.batch_sz:size(X_train, 1)
				# Adjusted the batch range to avoid out-of-bounds errors
				batch_end = min(i + nn.batch_sz - 1, size(X_train, 1))
				X_batch = Tensor(X_train[i:batch_end, :])
				Y_batch = Y_train[i:batch_end]

				# Adjust the size of Y_batch_encoded dynamically for the last batch
				batch_size_actual = size(X_batch, 1)

				Y_batch_encoded = zeros(batch_size_actual, nn.num_classes)
				for batch_index in 1:batch_size_actual
					class_index = Int(Y_batch[batch_index]) + 1
					if class_index > nn.num_classes || class_index < 1
						error("Classe non valida: $(Y_batch[batch_index]) per num_classes=$(nn.num_classes)")
					end
					Y_batch_encoded[batch_index, class_index] = 1
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
				err = loss.data[1]

				# Print verbose output
				if verbose && run % 10 == 0
					println("[$(epoch)/$(nn.epochs)] Loss: $(loss.data[1])")
				end
				run += 1
			end
		end
		return err
	end

	function evaluate(nn::NeuralNetwork, X_test::Matrix{<:Number}, Y_test::AbstractArray{<:Any}, score::Score)
		correct = 0
		total = 0
		for i in axes(Y_test, 1)
			X_in = X_test[i:i,:]
			X_in = Tensor(X_in)
			Y_true = Y_test[i]
			layer = nn.activation_functions[1](X_in * nn.layers[1] + nn.biases[1])
			for i in 2:size(nn.layers, 1)
				layer = nn.activation_functions[i](layer * nn.layers[i] + nn.biases[i])
			end
			#println("($(layer.data) : $(argmax(layer.data, dims=2)[1][2] - 1) vs $(Int(Y_true)))")
			if score(layer, Y_true)
				correct += 1
			end
			total += 1
		end
		return correct / total
	end

	function grid_search(η::AbstractArray{<:Number}, α::AbstractArray{<:Number}, batch_sz::AbstractArray{<:Int64},
			epochs::AbstractArray{<:Int64}, layer_sizes::AbstractArray{<:Tuple{<:Number, <:Number}}, initialize_weights_function::Initializer, initialize_biases_function::Initializer,
			activation_functions::AbstractArray{Activation, 1}, loss::Loss, regularization_function::MomentumFunction,
			X_train::Matrix{<:Number}, Y_train::AbstractArray{<:Any}, X_test::Matrix{<:Number}, Y_test::AbstractArray{<:Any},
			score::Score; verbose::Bool=false)
		
		grid_params = Dict(
			:η => η,
			:α => α,
			:batch_sz => batch_sz,
			:epochs => epochs
		)

		keys_order = (:η, :α, :batch_sz, :epochs)
		combinations = Iterators.product([grid_params[k] for k in keys_order]...)
		best_score = -Inf
		best_params = nothing
		nn = nothing

		for (a, b, c, d) in combinations
			# Extract the values for each parameter
			if verbose
				print("η: $(a), α: $(b), batch_size: $(c), epochs: $(d)...")
			end
			nn_prototype = NeuralNetwork(
					a,
					b,
					c,
					d,
					2, # num_classes
					layer_sizes,
					initialize_weights_function,
					initialize_biases_function,
					activation_functions,
					loss,
					regularization_function
			)
			err_scores = k_fold_cross_validation(nn_prototype, X_train, Y_train, score)
			valuation = evaluate(nn_prototype, X_test, Y_test, score)
			if verbose
				println(" $(valuation)% [avg err $(mean(err_scores))]")
			end
			if valuation > best_score
				best_score = valuation
				best_params = (a, b, c, d)
			end
		end

		return (nn, best_score, best_params)

	end

	function k_fold_cross_validation(nn::NeuralNetwork, X::Matrix{<:Number}, Y::AbstractArray{<:Any}, score::Score, k=5)
		n = length(Y)
		fold_size = div(n, k)
		indices = shuffle(1:n)  # Shuffle indices to randomize folds
		err_scores = []
	
		for i in 1:k
			# Split data into training and validation sets
			val_start = (i-1) * fold_size + 1
			val_end = i == k ? n : i * div(n, k)

			val_indices = indices[val_start:val_end]
			train_indices = setdiff(1:n, val_indices)
	
			X_train, y_train = X[train_indices, :], Y[train_indices]
			X_val, y_val = X[val_indices, :], Y[val_indices]
	
			# Train the model
			train(nn, X_train, y_train)
			err = evaluate(nn, X_val, y_val, score)


			push!(err_scores, err)
		end
	
		return err_scores
	end

end