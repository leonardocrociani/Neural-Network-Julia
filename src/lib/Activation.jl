module Activations

	using Random

	using FunctionWrappers

	export Activation
	using ..Tensors
	export tanh, relu, identity # activation functions

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

	# tanh:
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(tanh), ArgTypes}
		# Derivata di tanh: 1 - tanh^2(x)
		input_tensor = tensor.op.args[1]
		input_tensor.grad += (1 .- tensor.data .^ 2) .* tensor.grad
	end


	# relu
	function inner_relu(a::Tensor)
		# . is used for element-wise opereration in arrays. => .* is dot product, * is product
		return Tensor(max.(0,a.data), zeros(Float64, size(a.data)), Operation(relu, (a,)))
	end

	function relu()
		return Activation(relu)
	end

	# relu:
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(relu), ArgTypes}
		tensor.op.args[1].grad += (tensor.op.args[1].data .> 0) .* tensor.grad
	end

	# identity
	function inner_identity(tensor::Tensor)
		return tensor
	end

	function Activations.identity()
		return Activation(inner_identity)
	end

	# identity
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(identity), ArgTypes}
	end

end