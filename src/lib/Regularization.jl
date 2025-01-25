module Regularizations

	mutable struct MomentumFunction <: Function
		func::FunctionWrappers.FunctionWrapper{Number, Tuple{Tensor, Number, Number}}

		function MomentumFunction(func::Function)
			wrapped_func = FunctionWrappers.FunctionWrapper{Number, Tuple{Tensor, Number, Number}}(func)
			new(wrapped_func)
		end
	end

	# Make the struct callable
	function (wf::MomentumFunction)(tensor::Tensor, η::Number, α::Number)
		wf.func(tensor, η, α)
	end

	function inner_momentum(tensor::Tensor, η::Number, α::Number)
		return tensor
	end

	function momentum()
		return MomentumFunction(inner_momentum(tensor, η, α))
	end

	function inner_momentum_tikhonov(tensor::Tensor, η::Number, α::Number)
		return tensor
	end

	function momentum_tikhonov()
		return MomentumFunction(inner_momentum_tikhonov(tensor, η, α))
	end

end