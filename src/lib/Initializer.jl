module Initializers

	using Pkg
	using Random

	Pkg.add("FunctionWrappers")
	using FunctionWrappers

	export Initializer
	export random_initializer
	export zeros_initializer
	export ones_initializer

	mutable struct Initializer <: Function
		func::FunctionWrappers.FunctionWrapper{AbstractArray{<:Number}, Tuple{Number, Number}}

		function Initializer(func::Function)
			wrapped_func = FunctionWrappers.FunctionWrapper{AbstractArray{<:Number}, Tuple{Number, Number}}(func)
			new(wrapped_func)
		end
	end

	# Make the struct callable
	function (wf::Initializer)(x::Number, y::Number)
		wf.func(x, y)
	end

	# This seeds the RNG!!
	function random_initializer(seed::Number)
		Random.seed!(seed)
		# Return an Initializer that generates a random matrix with dimensions (x, y)
		return Initializer((x, y) -> Random.rand(Int(x), Int(y)))
	end

	# not seeding RNG!!
	function random_initializer()
		# Return an Initializer that generates a random matrix with dimensions (x, y)
		return Initializer((x, y) -> Random.rand(Int(x), Int(y)))
	end

	function zeros_initializer()
		return Initializer((x, y) -> zeros(Int(x), Int(y)))
	end

	function ones_initializer()
		return Initializer((x, y) -> ones(Int(x), Int(y)))
	end

end