module DataTypes



mutable struct Value{opType} <: Number  # Value type is a subtype of Number
    data::Float64   # the actual value stored 
    grad::Float64   # the value that will be computed after a backward() operation (later)
    op::opType      # used to track the operation from which the Value is created (definition, operation...)
end

struct Operation{FuncType, ArgTypes} # to track the type of operation (addition, subtraction, ...) (argtype: type of the operands)
    op::FuncType
    args::ArgTypes
end

# now the constructors:
# Value from definition:
Value(x::Number) = Value(Float64(x), 0.0, nothing) # grad is 0.0 for now, op is nothing for a definition

export Value
export Operation

# for printing the values in a nice way:
import Base.show
Base.show(io::IO, v::Value) = print(io, "Value($(v.data))")

# overriding the == comparison so that it returns true only if the 2 Value objects are actually the same (in memory)
import Base.==
function ==(a::Value, b::Value)
    return a===b
end

# overriding the + operator so that it returns the sum of two values as a new value
import Base.+
function +(a::Value, b::Value)
	# data is simply the sum of the data inside the pair of values
	# we are using 0.0 as a placeholder value for the gradient
	# we store the addition as an Operation of FuncType +, ArgType is the pair of values (a, b)
	return Value(a.data + b.data, 0.0, Operation(+, (a, b)))
end

# backprop function for the case in which the val parameter comes from an addition operation
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}
	# let val = a + b, our aim is to update:
	# a.grad, b.grad
	# we use the rule for the derivative of the sum:
	val.op.args[1].grad += val.grad
	val.op.args[2].grad += val.grad
end

end

# backprop! is an internal function that updates the gradients for the operands 
# depending on the operation used to create the variable.

# The user calls the `backward` function on the final resulting variable for which 
# they want to calculate the derivative. This function handles the correct 
# backpropagation for the operation that was used to create the variable.

# We perform a topological sort using DFS. We start from 
# the final variable and update the gradients for the operands of the operation 
# that was used to create the variable. At the end, we want an array where each 
# value is preceded by all the operands that were used to produce it.

function backward(a::Value)
	function build_topo(v::Value, visited=Value[], topo=Value[])
		if !(v in visited)
			push!(visited, v)
			if v.op != nothing
				for operand in v.op.args
					if operand isa Value
						build_topo(operand, visited, topo)
					end
				end
			end
			push!(topo, v)
		end
		return topo
	end
	topo = build_topo(a)
	a.grad = 1
	for node in reverse(topo)
		backprop!(node)
	end
end
## adding support for operation Value + number
Base.promote_rule(::Type{<:Value}, ::Type{T}) where {T<:Number} = Value