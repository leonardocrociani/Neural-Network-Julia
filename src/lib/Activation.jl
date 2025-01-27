module Activations

	using Random
	using FunctionWrappers
	export Activation
	using ..Tensors
	export tanh, relu, identity

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

	# relu function
	function inner_relu(a::Tensor)
		return Tensor(max.(0,a.data), zeros(Float64, size(a.data)), Operation(relu, (a,)))
	end

	function relu()
		return Activation(relu)
	end

	# relu backpropagation
	function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(relu), ArgTypes}
		tensor.op.args[1].grad += (tensor.op.args[1].data .> 0) .* tensor.grad
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

	function sigmoid(x)
        return 1 / (1 + exp(-x))
    end

    # Then use it in inner_sigmoid
    function inner_sigmoid(a::Tensor)
        out_data = 1 ./(1 .+ exp.(-a.data))
        return Tensor(out_data, zeros(Float64, size(out_data)), Operation(sigmoid, (a,)))
    end

    function Activations.sigmoid()
        return Activation(inner_sigmoid)
    end

    # sigmoid backpropagation
    function Tensors.backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(sigmoid), ArgTypes}
        input_tensor = tensor.op.args[1]
        input_tensor.grad += tensor.data .* (1 .- tensor.data) .* tensor.grad
    end


end
