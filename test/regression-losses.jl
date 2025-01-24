include("../src/lib/Tensor.jl") 
using .Tensors 

function test_regression_losses()
    # tested in python herer: https://colab.research.google.com/drive/18T0rKt-uNPv7b6HDEnVlXl-mOsuI_ajU#scrollTo=L4yts73e4c4n
    y_true = [1.0 2.0 3.0; 4.0 5.0 6.0]
    y_pred = Tensor([1.5 2.5 3.5; 4.5 5.5 6.5])

    println("Mean Squared Error (MSE): ", Tensors.mse(y_pred, y_true).data[1])
    println("Mean Euclidean Error: ", Tensors.mean_euclidean_error(y_pred, y_true).data[1])
    println("Root Mean Squared Error (RMSE): ", Tensors.rmse(y_pred, y_true).data[1])

    # now passing grad=true

    tensor = Tensors.mse(y_pred, y_true, grad=true)
    println("MSE grad: ", y_pred.grad)

    tensor = Tensors.mean_euclidean_error(y_pred, y_true, grad=true)
    println("Mean Euclidean Error grad: ", y_pred.grad)

    tensor = Tensors.rmse(y_pred, y_true, grad=true)
    println("RMSE grad: ", y_pred.grad)
end

test_regression_losses()