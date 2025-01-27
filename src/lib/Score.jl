module Scores

	using Random
	using FunctionWrappers
	export Score
	using ..Tensors
	export monk

	mutable struct Score <: Function
		func::FunctionWrappers.FunctionWrapper{Bool, Tuple{Any, Any}}

		function Score(func::Function)
			wrapped_func = FunctionWrappers.FunctionWrapper{Bool, Tuple{Any, Any}}(func)
			new(wrapped_func)
		end
	end

	# Make the struct callable
	function (wf::Score)(a::Tensor, b::Any)
		wf.func(a, b)
	end

	function inner_monk(a::Tensor, b::Any)
		return argmax(a.data, dims=2)[1][2] - 1 == Int(b)
	end

	function monk()
		return Score(inner_monk)
	end
end