module Tensors

	export Tensor
	export Operation
	export backward

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

end