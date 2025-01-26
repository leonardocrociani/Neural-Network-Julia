module Regularizations

	using Pkg
	using Random

	Pkg.add("FunctionWrappers")
	using FunctionWrappers

	export MomentumFunction
	using ..Tensors

	export momentum
	export momentum_tikhonov

	mutable struct MomentumFunction <: Function
		func::FunctionWrappers.FunctionWrapper{Nothing, Tuple{Tensor, AbstractArray{Tensor, 1}, Vector{Matrix{Float64}}, Number, Number}}

		function MomentumFunction(func::Function)
			wrapped_func = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Tensor, AbstractArray{Tensor, 1}, Vector{Matrix{Float64}}, Number, Number}}(func)
			new(wrapped_func)
		end
	end

	# Make the struct callable
	function (wf::MomentumFunction)(loss::Tensor, layers::AbstractArray{Tensor, 1}, Δw_old::Vector{Matrix{Float64}}, η::Number, α::Number)
		wf.func(loss, layers, Δw_old, η, α)
		return nothing
	end

	function inner_momentum(loss::Tensor, layers::AbstractArray{Tensor, 1}, Δw_old::Vector{Matrix{Float64}}, η::Number, α::Number)
		Tensors.backward(loss)
		for i in eachindex(layers)
			layer = layers[i]
			Δw_new = layer.grad .* η - α .* Δw_old[i]
			Δw_old[i] = Δw_new
			layer.data = layer.data - Δw_new
		end
		return nothing
	end

	function momentum()
		return MomentumFunction(inner_momentum)
	end

	function inner_momentum_tikhonov(loss::Tensor, layers::AbstractArray{Tensor, 1}, Δw_old::Vector{Matrix{Float64}}, η::Number, α::Number)
		prev_layers = []
		for i in eachindex(layers)
			layer = layers[i]
			push!(prev_layers, deepcopy(layer.data))
			layer.data = layer.data + Δw_old[i] .* α
		end
		Tensors.backward(loss)
		for i in eachindex(layers)
			layer = layers[i]
			Δw_new = (-η) .* layer.grad + α .* Δw_old[i]
			Δw_old[i] = Δw_new
			layer.data = prev_layers[i] + Δw_new
		end
		return nothing
	end


	function momentum_tikhonov()
		return MomentumFunction(inner_momentum_tikhonov)
	end

end