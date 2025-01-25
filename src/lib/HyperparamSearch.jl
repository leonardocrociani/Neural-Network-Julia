# Include the Metrics.jl file
include("./Metrics.jl")

module HyperparamSearch

    using ..Tensors
    using ..NeuralNetworks
    using ..Metrics: accuracy
    using Random

    export grid_search, random_search
    function train_model(
        nn::NeuralNetwork,
        X_train::Matrix{<:Number},
        Y_train::AbstractArray{<:Any};
        X_val::Union{Matrix{<:Number}, Nothing} = nothing,
        Y_val::Union{AbstractArray{<:Any}, Nothing} = nothing
    )
        train(nn, X_train, Y_train, verbose=false)
    
        if X_val !== nothing && Y_val !== nothing
            num_samples = size(X_val, 1)
            predictions = Vector{Int}(undef, num_samples)
            
            for i in 1:num_samples
                X_input = Tensor(reshape(Float64.(X_val[i, :]), 1, :))
                output = X_input * nn.layers[1] + nn.biases[1]
                for layer_idx in 2:length(nn.layers)
                    output = nn.activation_functions[layer_idx](
                        output * nn.layers[layer_idx] + nn.biases[layer_idx]
                    )
                end
                predictions[i] = argmax(output.data, dims=2)[1][2] - 1
            end
    
            predictions_2d = reshape(predictions, (length(predictions), 1))
            y_val_2d = reshape(Int.(Y_val), (length(Y_val), 1))  # Ensure y_val_2d is Array{Int, 2}
            return accuracy(Tensor(Float64.(predictions_2d)), y_val_2d)  # Convert to Float64
        else
            return 0.0
        end
    end
    function grid_search(
        nn_prototype::NeuralNetwork,
        X_train::Matrix{<:Number},
        Y_train::AbstractArray{<:Any},
        X_val::Matrix{<:Number},
        Y_val::AbstractArray{<:Any},
        param_grid::Dict{Symbol, Vector}
    )
        best_score = -Inf
        best_params = nothing

        param_keys = collect(keys(param_grid))
        function recurse(idx, current_params)
            if idx > length(param_keys)
                 # here we copy the NN prototype and set the fields to the current params ( deep copy as we dont want a reference)
                nn = deepcopy(nn_prototype)
                for (k, v) in current_params
                    setfield!(nn, k, v)
                end

                score = train_model(nn, X_train, Y_train; X_val=X_val, Y_val=Y_val)
                if score > best_score
                    best_score = score
                    best_params = deepcopy(current_params)
                end
                return
            end
            for val in param_grid[param_keys[idx]]
                current_params[param_keys[idx]] = val
                recurse(idx + 1, current_params)
            end
        end

        recurse(1, Dict{Symbol, Any}())
        return best_params, best_score
    end

    function random_search(
        nn_prototype::NeuralNetwork,
        X_train::Matrix{<:Number},
        Y_train::AbstractArray{<:Any},
        X_val::Matrix{<:Number},
        Y_val::AbstractArray{<:Any},
        param_ranges::Dict{Symbol, Vector},
        n_iters::Int
    )
        best_score = -Inf
        best_params = nothing

        for _ in 1:n_iters
            current_params = Dict{Symbol, Any}()
            for (k, v) in param_ranges
                current_params[k] = rand(v)
            end
 
            nn = deepcopy(nn_prototype)
            for (k, v) in current_params
                setfield!(nn, k, v)
            end

            score = train_model(nn, X_train, Y_train; X_val=X_val, Y_val=Y_val)
            if score > best_score
                best_score = score
                best_params = deepcopy(current_params)
            end
        end

        return best_params, best_score
    end

end