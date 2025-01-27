module Activations

	using Random
	using FunctionWrappers
	export Activation
	using ..Tensors
	export tanh, identity

	mutable struct Activation <: Function
		func::FunctionWrappers.FunctionWrapper{Tensor, Tuple{Tensor}}

		function Activation(func::Function)
			wrapped_func = FunctionWrappers.FunctionWrapper{Tensor, Tuple{Tensor}}(func)
			new(wrapped_func)
		end
	end

	# Make the struct callable
	function (wf::Activation)(tensor::Tensor)
		wf.func(tensor)
	end

	# tanh function
	import Base.tanh
	function inner_tanh(a::Tensor)
		# Calcolo della funzione tanh
		out_data = tanh.(a.data)
		# Creazione di un nuovo Tensor
		return Tensor(out_data, zeros(Float64, size(out_data)), Operation(tanh, (a,)))
	end

	function Activations.tanh()
		return Activation(inner_tanh)
	end

	# tanh backpropagation
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(tanh), ArgTypes}
		# Derivata di tanh: 1 - tanh^2(x)
		input_tensor = tensor.op.args[1]
		input_tensor.grad += (1 .- tensor.data .^ 2) .* tensor.grad
	end

	# identity function
	function inner_identity(tensor::Tensor)
		return tensor
	end

	function Activations.identity()
		return Activation(inner_identity)
	end

	# identity backpropagation
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(identity), ArgTypes}
	end

	relu(x) = max(0, x)

	function inner_relu(a::Tensor)
		# Calculate ReLU function
		out_data = max.(0, a.data)
		# Create new Tensor
		return Tensor(out_data, zeros(Float64, size(out_data)), Operation(relu, (a,)))
	end

	function Activations.relu()
		return Activation(inner_relu)
	end

	# relu backpropagation
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(relu), ArgTypes}
		# Derivative of ReLU: 1 if x > 0, 0 otherwise
		input_tensor = tensor.op.args[1]
		input_tensor.grad += (tensor.data .> 0) .* tensor.grad
	end

	# Define sigmoid function
	sigmoid(x) = 1 / (1 + exp(-x))

	function inner_sigmoid(a::Tensor)
		# Calculate sigmoid function
		out_data = sigmoid.(a.data)
		# Create new Tensor
		return Tensor(out_data, zeros(Float64, size(out_data)), Operation(sigmoid, (a,)))
	end

	function Activations.sigmoid()
		return Activation(inner_sigmoid)
	end

	# sigmoid backpropagation
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(sigmoid), ArgTypes}
		# Derivative of sigmoid: σ(x)(1 - σ(x))
		input_tensor = tensor.op.args[1]
		input_tensor.grad += tensor.data .* (1 .- tensor.data) .* tensor.grad
	end
end
