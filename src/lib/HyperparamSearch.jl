# Include the Metrics.jl file
include("./Metrics.jl")

module HyperparamSearch

    using Statistics
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
        X::Matrix{<:Number},
        Y::AbstractArray{<:Any},
        param_grid::Dict{Symbol, Vector};
        X_val::Union{Matrix{<:Number}, Nothing} = nothing,
        Y_val::Union{AbstractArray{<:Any}, Nothing} = nothing,
        cv::Bool = false,
        k::Int = 5,
        verbose::Bool = false
    )
        best_score = -Inf
        best_params = nothing
    
        param_keys = collect(keys(param_grid))
        function recurse(idx, current_params)
            if idx > length(param_keys)
                nn = deepcopy(nn_prototype)
                for (k, v) in current_params
                    setfield!(nn, k, v)
                end
    
                if cv
                    n_samples = size(X, 1)
                    fold_size = div(n_samples, k)
                    scores = Vector{Float64}()
    
                    if verbose
                        println("Starting cross-validation with $k folds...")
                        println("Dataset size: $n_samples rows, fold size: $fold_size")
                    end

                    for i in 1:k
                        val_start = (i - 1) * fold_size + 1
                        val_end = min(i * fold_size, n_samples) 
                        val_indices = val_start:val_end
                    
                        if verbose
                            println("Fold $i:")
                            println("  Validation indices: $val_start to $val_end")
                        end
                        
                        X_val_fold = nothing
                        Y_val_fold = nothing
                        try
                            X_val_fold = X[val_indices, :]
                            Y_val_fold = Y[val_indices]
                        catch e
                            println("ERROR: Validation indices out of range.")
                            println("  val_start: $val_start, val_end: $val_end, dataset size: $n_samples")
                            rethrow(e) 
                        end
                    
                        train_indices = setdiff(1:n_samples, val_indices)
                        if verbose
                            println("  Training indices: $(train_indices[1]) to $(train_indices[end]) (total: $(length(train_indices)))")
                        end

                        X_train_fold = X[train_indices, :]
                        Y_train_fold = Y[train_indices]
                    
                        if verbose
                            println("  Training model for fold $i...")
                        end
                        score = train_model(nn, X_train_fold, Y_train_fold; X_val=X_val_fold, Y_val=Y_val_fold)
                        if verbose 
                            println("  Score for fold $i: $score")
                        end
                        push!(scores, score)
                    end
                    
    
                    avg_score = mean(scores)
                    if verbose
                        println("Average score across folds: $avg_score")
                    end
                    if avg_score > best_score

                        if verbose
                            println("  New best score: $avg_score, updating best parameters...")
                        end
                        best_score = avg_score
                        best_params = deepcopy(current_params)
                    end
                else
                    if X_val === nothing || Y_val === nothing
                        error("Validation set must be provided when cv=false.")
                    end
                    if verbose
                        println("Training model with validation set...")
                    end
                    score = train_model(nn, X, Y; X_val=X_val, Y_val=Y_val)
                    if verbose
                        println("Validation score: $score")
                    end
                    if score > best_score
                        if verbose
                            println("  New best score: $score, updating best parameters...")
                        end
                        best_score = score
                        best_params = deepcopy(current_params)
                    end
                end
                return
            end
    
            for val in param_grid[param_keys[idx]]
                current_params[param_keys[idx]] = val
                if verbose
                    println("Trying parameter: $(param_keys[idx]) = $val")
                end
                recurse(idx + 1, current_params)
            end
        end
    
        if cv && k > size(X, 1)
            error("Number of folds (k) cannot exceed the number of samples.")
        end
    
        recurse(1, Dict{Symbol, Any}())
        return best_params, best_score
    end

    function random_search(
        nn_prototype::NeuralNetwork,
        X::Matrix{<:Number},
        Y::AbstractArray{<:Any},
        param_ranges::Dict{Symbol, Vector},
        n_iters::Int;
        X_val::Union{Matrix{<:Number}, Nothing} = nothing,
        Y_val::Union{AbstractArray{<:Any}, Nothing} = nothing,
        cv::Bool = false,
        k::Int = 5, 
        verbose::Bool = false
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

            if cv
                n_samples = size(X, 1)
                fold_size = div(n_samples, k)
                scores = Vector{Float64}()
    
                if verbose
                    println("Starting cross-validation with $k folds...")
                    println("Dataset size: $n_samples rows, fold size: $fold_size")
                end
    
                for i in 1:k
                    val_start = (i - 1) * fold_size + 1
                    val_end = min(i * fold_size, n_samples)  # Ensure val_end doesn't exceed bounds
                    val_indices = val_start:val_end
                    
                    if verbose
                        println("Fold $i:")
                        println("  Validation indices: $val_start to $val_end")
                    end
                    
                    X_val_fold = nothing
                    Y_val_fold = nothing
                    try
                        X_val_fold = X[val_indices, :]
                        Y_val_fold = Y[val_indices]
                    catch e
                        println("ERROR: Validation indices out of range.")
                        println("  val_start: $val_start, val_end: $val_end, dataset size: $n_samples")
                        rethrow(e) 
                    end
                    
                    train_indices = setdiff(1:n_samples, val_indices)
                    if verbose
                        println("  Training indices: $(train_indices[1]) to $(train_indices[end]) (total: $(length(train_indices)))")
                    end
                    
                    X_train_fold = X[train_indices, :]
                    Y_train_fold = Y[train_indices]
                    
                    if verbose
                        println("  Training model for fold $i...")
                    end
                    score = train_model(nn, X_train_fold, Y_train_fold; X_val=X_val_fold, Y_val=Y_val_fold)
                    if verbose
                        println("  Score for fold $i: $score")
                    end
                    push!(scores, score)
                end

                avg_score = mean(scores)
                if verbose
                    println("Average score across folds: $avg_score")
                end
                if avg_score > best_score
                    if verbose
                        println("  New best score: $avg_score, updating best parameters...")
                    end
                    best_score = avg_score
                    best_params = deepcopy(current_params)
                end
            else
                if X_val === nothing || Y_val === nothing
                    error("Validation set must be provided when cv=false.")
                end
                if verbose
                    println("Training model with validation set...")
                end
                score = train_model(nn, X, Y; X_val=X_val, Y_val=Y_val)
                if verbose
                    println("Validation score: $score")
                end
                if score > best_score
                    if verbose
                        println("  New best score: $score, updating best parameters...")
                    end
                    best_score = score
                    best_params = deepcopy(current_params)
                end
            end
        end

        return best_params, best_score
    end


end