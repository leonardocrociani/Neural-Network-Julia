module PlotLearningCurve

using Plots

export plot_learning_curve

function plot_learning_curve(accuracy::Vector{Float64}, loss::Union{Vector{Float64}, Nothing}=nothing)
    epochs = 1:length(accuracy)
    plot(epochs, accuracy, label="Training Accuracy", xlabel="Epochs", ylabel="Value", title="Learning Curve")
    if loss !== nothing
        plot!(epochs, loss, label="Training Loss")
    end
end

end