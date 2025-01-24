module Tensors

	using Statistics

	export Tensor
	export Operation
	export backward
	export relu, tanh # funzioni di attivazione
	export softmax_crossentropy, mse, mean_euclidean_error, rmse # losses

	# ================================ Basic Tensor type ================================ #

	mutable struct Tensor{opType} <: AbstractArray{Float64, 2}  # Tensor type is a subtype of AbstractArray
		# Using 2-dimensional arrays of Float64 objects
		data::Array{Float64, 2}   # the actual value stored 
		grad::Array{Float64, 2}   # the values that will be computed after a backward() operation (later)
		op::opType      # used to track the operation from which the Value is created (definition, operation...)
	end

	struct Operation{FuncType, ArgTypes} # to track the type of operation (addition, subtraction, ...) (argtype: type of the operands)
		op::FuncType
		args::ArgTypes
	end

	# now the constructors
	# Tensors are defined as:
	Tensor(x::Array{Float64, 2}) = Tensor(x, zeros(Float64, size(x)), nothing) # grad is an array of zeros for now, op is nothing by definition

	# We can also create tensors from row or column vectors
	# `column_vector` is an optional parameter, it is toggled on if x is a column vector
	function Tensor(x::Array{Float64, 1}; column_vector::Bool=false)
		# sizes are indicated as (rows, columns)
		if column_vector # we reshape x to size (N, 1) with N size of x
			data_2d = reshape(x, (length(x), 1))
		else # we have a row vector, we reshape x to size (1, N)
			data_2d = reshape(x, (1, length(x)))
		end
		
		return Tensor(data_2d, zeros(Float64, size(data_2d)), nothing)
	end

	# For printing tensors in a nice way
	import Base.show
	Base.show(io::IO, tensor::Tensor) = print(io, "Tensor($(tensor.data))")

	# If the tensor was defined by the user and not obtained by an operation, do not backpropagate
	backprop!(val::Tensor{Nothing}) = nothing

	# Overriding == comparison for tensors
	import Base.==
	function ==(a::Tensor, b::Tensor)
		return a === b
	end

	## ================================ Tensor type operations ================================ ##

	Base.size(x::Tensor) = size(x.data) # print the size of a Tensor by calling the size function on it
	Base.getindex(x::Tensor, i...) = getindex(x.data, i...) # needed for indexing so instead of writing x.data[3,4] we can omit .data to access the values
	Base.setindex!(x::Tensor, v, i...) = setindex!(x.data, v, i...) # same as before but for setting values

	import Base.* # overriding the * operator so that it returns the product of two tensors as a new tensor ( matrix multiplication )
	function *(a::Tensor, b::Tensor)
		out = a.data * b.data
		return Tensor(out, zeros(Float64, size(out)), Operation(*, (a, b)))
	end

	import Base.+ # overriding the + operator so that it returns the sum of two tensors as a new tensor
	function +(a::Tensor, b::Tensor)
		out = a.data .+ b.data
		# Brocasting happens automatically in case of row-vector
		# data is simply the sum of the data inside the pair of tensors
		# we are using 0.0 as a placeholder value for the gradient
		return Tensor(out, zeros(Float64, size(out)), Operation(+, (a, b)))
	end

	# relu
	function relu(a::Tensor)
		# . is used for element-wise opereration in arrays. => .* is dot product, * is product
		return Tensor(max.(0,a.data), zeros(Float64, size(a.data)), Operation(relu, (a,)))
	end

	# tanh function
	import Base.tanh
	function tanh(a::Tensor)
		# Calcolo della funzione tanh
		out_data = tanh.(a.data)
		# Creazione di un nuovo Tensor
		return Tensor(out_data, zeros(Float64, size(out_data)), Operation(tanh, (a,)))
	end

	# softmax crossentropy
	# y_true is the one hot encoded true labels!
	function softmax_crossentropy(a::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}}; grad::Bool = false) # grad will be used later 
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

		if grad
			samples = size(probs, 1)
			a.grad = copy(probs)
			argmax_y_true = argmax(y_true, dims=2)

			for sample_index in 1:samples
				a.grad[sample_index, argmax_y_true[sample_index][2]] -= 1
			end

			a.grad = a.grad ./ samples
		end

		out = reshape(out, (1, 1))

		return Tensor(out, zeros(Float64, size(out)), Operation(softmax_crossentropy, (a,)))
	end

	# mean squared error
	function mse(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}}; grad::Bool=false)
		loss = mean((y_pred.data .- y_true) .^ 2)
		out = [loss]
		if grad
			y_pred.grad = 2 * (y_pred.data .- y_true) / length(y_pred.data)
		end
		out = reshape(out, (1, 1))
		return Tensor(out, zeros(Float64, size(out)), Operation(mse, (y_pred, y_true)))
	end

	# mean euclidean error
	function mean_euclidean_error(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}}; grad::Bool=false)
		loss = mean(sqrt.(sum((y_pred.data .- y_true) .^ 2, dims=2)))
		out = [loss]
		if grad
			diff = y_pred.data .- y_true
			norms = sqrt.(sum(diff .^ 2, dims=2))
			y_pred.grad = diff ./ (norms .+ 1e-7) ./ size(y_pred.data, 1)  # Evita divisione per zero
		end
		out = reshape(out, (1, 1))
		return Tensor(out, zeros(Float64, size(out)), Operation(mean_euclidean_error, (y_pred, y_true)))
	end

	# root mean squared error
	function rmse(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64,2}}; grad::Bool=false)
		loss = sqrt(mean((y_pred.data .- y_true) .^ 2))
		out = [loss]
		if grad
			diff = y_pred.data .- y_true
			loss = sqrt(mean(diff .^ 2))
			y_pred.grad = diff ./ (loss .+ 1e-7) ./ length(y_pred.data)  # Evita divisione per zero
		end
		out = reshape(out, (1, 1))
		return Tensor(out, zeros(Float64, size(out)), Operation(rmse, (y_pred, y_true)))
	end

	# multiplication
	function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(*), ArgTypes}
		# tensor = a + backprop
		# backprop! = (tensor)
		# udpate a.gard, b.grad

		tensor.op.args[1].grad += tensor.grad * transpose(tensor.op.args[2].data) 
		tensor.op.args[2].grad += transpose(tensor.op.args[1].data) * tensor.grad
	end

	# backprop function for the case in which the val parameter comes from a sum
	function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}
		# update a.grad b.grad, a and b are 2-dimensional a
		# Here we basically do a reverse broadcast based on the size of the gradient 
		
		if size(tensor.grad) == size(tensor.op.args[1].data)
			tensor.op.args[1].grad += ones(size(tensor.op.args[1].data)) .* tensor.grad
		else
			tensor.op.args[1].grad += ones(size(tensor.op.args[1].grad)) .* sum(tensor.grad, dims=1) # Reverse broadcast
		end
		
		if size(tensor.grad) == size(tensor.op.args[2].data)
			tensor.op.args[2].grad += ones(size(tensor.op.args[2].data)) .* tensor.grad
		else
			tensor.op.args[2].grad += ones(size(tensor.op.args[2].grad)) .* sum(tensor.grad, dims=1) # Reverse broadcast
		end
	end

	# relu:
	function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(relu), ArgTypes}
		tensor.op.args[1].grad += (tensor.op.args[1].data .> 0) .* tensor.grad
	end

	# tanh:
	function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(tanh), ArgTypes}
		# Derivata di tanh: 1 - tanh^2(x)
		input_tensor = tensor.op.args[1]
		input_tensor.grad += (1 .- tensor.data .^ 2) .* tensor.grad
	end

	# backprop in case of `softmax_crossentropy`
	function backprop!(_tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(softmax_crossentropy), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the `softmax_crossentropy` function
	end

	# backprop in case of `mse`
	function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(mse), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the mse function
		# tensor.op.args[1].grad += 2 * (tensor.op.args[1].data .- tensor.op.args[2]) / length(tensor.op.args[1].data)
	end

	# backprop in case of `mean_euclidean_error`
	function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(mean_euclidean_error), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the mean_euclidean_error function
		# tensor.op.args[1].grad += (tensor.op.args[1].data .- tensor.op.args[2]) ./ length(tensor.op.args[1].data)
	end

	# backprop in case of `rmse`
	function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(rmse), ArgTypes}
		# it should do nothing, we have implemented the backprop step inside the rmse function
		# loss = sqrt(mean((tensor.op.args[1].data .- tensor.op.args[2]) .^ 2))
		# tensor.op.args[1].grad += (tensor.op.args[1].data .- tensor.op.args[2]) / (length(tensor.op.args[1].data) * loss)
	end


	function backward(a::Tensor)

		function build_topo(v::Tensor, visited=Tensor[], topo=Tensor[])
			if !(v in visited)
				push!(visited, v)

				if !isnothing(v.op)
					for operand in v.op.args
						if operand isa Tensor
							build_topo(operand, visited, topo)
						end
					end
				end

				push!(topo, v)
			end

			return topo
		end
		
		topo = build_topo(a)
		a.grad .= 1 # .= is used to broadcast the value 1 to all the elements of the array, here is neede as we are working with matrices	

		for node in reverse(topo)
			backprop!(node)
		end

	end


end # this is the END of the module, should be the last instruction!

