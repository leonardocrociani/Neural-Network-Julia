module Metrics

    using ..Tensors
    using Statistics

    export accuracy, precision, recall, f1_score

    function accuracy(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64, 2}})
        # Convert y_true to a 1-dimensional array if it's 2D (assuming 1D for simplicity)
        y_true = vec(y_true)
        
        # Ensure y_pred is also a 1D array
        y_pred = vec(y_pred.data)

        correct = 0
        for i in 1:length(y_true)
            correct += y_pred[i] == y_true[i]
        end

        return correct / length(y_true)
    end

    function precision(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64, 2}})
        # Convert y_true to a 1-dimensional array if it's 2D (assuming 1D for simplicity)
        y_true = vec(y_true)
        
        # Ensure y_pred is also a 1D array
        y_pred = vec(y_pred.data)

        true_positives = 0
        false_positives = 0
        for i in 1:length(y_true)
            true_positives += y_pred[i] == y_true[i] && y_pred[i] == 1
            false_positives += y_pred[i] != y_true[i] && y_pred[i] == 1
        end

        if true_positives + false_positives == 0
            return 0.0
        end

        return true_positives / (true_positives + false_positives)
    end


    function recall(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64, 2}})
        # Convert y_true to a 1-dimensional array if it's 2D (assuming 1D for simplicity)
        y_true = vec(y_true)
        
        # Ensure y_pred is also a 1D array
        y_pred = vec(y_pred.data)

        true_positives = 0
        false_negatives = 0
        for i in 1:length(y_true)
            true_positives += y_pred[i] == y_true[i] && y_pred[i] == 1
            false_negatives += y_pred[i] != y_true[i] && y_pred[i] == 0
        end

        return true_positives / (true_positives + false_negatives)
    end

    function f1_score(y_pred::Tensor, y_true::Union{Array{Int, 2}, Array{Float64, 2}})
        prec = precision(y_pred, y_true)
        rec = recall(y_pred, y_true)

        if prec + rec == 0
            return 0.0
        end

        return 2 * (prec * rec) / (prec + rec)
    end

end
