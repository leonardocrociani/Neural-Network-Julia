module Losses

	using Random

	using FunctionWrappers

	export Loss
	using ..Tensors
	export softmax_crossentropy, mse, mean_euclidean_error, rmse # losses

	mutable struct Loss <: Function
		func::FunctionWrappers.FunctionWrapper{Tensor, Tuple{Tensor, Union{Array{Int, 2}, Array{Float64,2}}}}

		function Loss(func::Function)
			wrapped_func = FunctionWrappers.FunctionWrapper{Tensor, Tuple{Tensor, Union{Array{Int, 2}, Array{Float64,2}}}}(func)
			new(wrapped_func)
		end
	end

	# Make the struct callable
	function (wf::Loss)(a::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}})
		wf.func(a, y_true)
	end

	# softmax crossentropy
	# y_true is the one hot encoded true labels!
	function inner_softmax_crossentropy(a::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}})
		# softmax
		exp_values = exp.(a.data .- maximum(a.data, dims=2)) # we subtract to avoid float overflow
		probs = exp_values ./ sum(exp_values, dims=2) # sum calculates row-wise sum
		probs_clipped = clamp.(probs, 1e-7, 1-1e-7)

		# the array of the probability of the correct answer for each sample
		correct_confidences = sum(probs_clipped .* y_true, dims=2)
		
		# negative log likelyhood
		sample_losses = -log.(correct_confidences)

		# we output the mean loss across the batch
		out = [sum(sample_losses) / length(sample_losses)]

		# SIDE EFFECT: update gradient
		samples = size(probs, 1)
		a.grad = copy(probs)
		argmax_y_true = argmax(y_true, dims=2)
		for sample_index in 1:samples
			a.grad[sample_index, argmax_y_true[sample_index][2]] -= 1
		end
		a.grad = a.grad ./ samples

		out = reshape(out, (1, 1))
		return Tensor(out, zeros(Float64, size(out)), Operation(softmax_crossentropy, (a,)))
	end

	function softmax_crossentropy()
		return Loss(inner_softmax_crossentropy)
	end

	# backprop in case of `softmax_crossentropy`
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(softmax_crossentropy), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the `softmax_crossentropy` function
	end


	# mean squared error
	function inner_mse(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}})
		loss = mean((y_pred.data .- y_true) .^ 2)
		out = [loss]

		y_pred.grad = 2 * (y_pred.data .- y_true) / length(y_pred.data)

		out = reshape(out, (1, 1))
		return Tensor(out, zeros(Float64, size(out)), Operation(mse, (y_pred, y_true)))
	end

	function mse()
		return Loss(inner_mse)
	end

	# backprop in case of `mse`
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(mse), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the mse function
		# tensor.op.args[1].grad += 2 * (tensor.op.args[1].data .- tensor.op.args[2]) / length(tensor.op.args[1].data)
	end


	# mean euclidean error
	function inner_mean_euclidean_error(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}})
		loss = mean(sqrt.(sum((y_pred.data .- y_true) .^ 2, dims=2)))
		out = [loss]

		diff = y_pred.data .- y_true
		norms = sqrt.(sum(diff .^ 2, dims=2))
		y_pred.grad = diff ./ (norms .+ 1e-7) ./ size(y_pred.data, 1)  # Evita divisione per zero

		out = reshape(out, (1, 1))
		return Tensor(out, zeros(Float64, size(out)), Operation(mean_euclidean_error, (y_pred, y_true)))
	end

	function mean_euclidean_error()
		return Loss(mean_euclidean_error)
	end

	# backprop in case of `mean_euclidean_error`
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(mean_euclidean_error), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the mean_euclidean_error function
		# tensor.op.args[1].grad += (tensor.op.args[1].data .- tensor.op.args[2]) ./ length(tensor.op.args[1].data)
	end


	# root mean squared error
	function inner_rmse(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}})
		loss = sqrt(mean((y_pred.data .- y_true) .^ 2))
		out = [loss]

		diff = y_pred.data .- y_true
		loss = sqrt(mean(diff .^ 2))
		y_pred.grad = diff ./ (loss .+ 1e-7) ./ length(y_pred.data)  # Evita divisione per zero

		out = reshape(out, (1, 1))
		return Tensor(out, zeros(Float64, size(out)), Operation(rmse, (y_pred, y_true)))
	end

	function rmse()
		return Loss(inner_rmse)
	end

	# backprop in case of `rmse`
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(rmse), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the rmse function
		# loss = sqrt(mean((tensor.op.args[1].data .- tensor.op.args[2]) .^ 2))
		# tensor.op.args[1].grad += (tensor.op.args[1].data .- tensor.op.args[2]) / (length(tensor.op.args[1].data) * loss)
	end


end