module DataTypes

export Value
export Operation

mutable struct Value{opType} <: Number  # Value type is a subtype of Number
    data::Float64   # the actual value stored 
    grad::Float64   # the value that will be computed after a backward() operation (later)
    op::opType      # used to track the operation from which the Value is created (definition, operation...)
end

struct Operation{FuncType, ArgType} # to track the type of operation (addition, subtraction, ...) (argtype: type of the operands)
    op::FuncType
    args::ArgType
end

# now the constructors:
# Value from definition:
Value(x::Number) = Value(Float64(x), 0.0, nothing) # grad is 0.0 for now, op is nothing for a definition


# for printing the values in a nice way:
import Base
Base.show(io::IO, v::Value) = print(io, "Value($(v.data))")

# overriding the == comparison so that it returns true only if the 2 Value objects are actually the same (in memory)
import Base.==
function ==(a::Value, b::Value)
    return a===b
end

end