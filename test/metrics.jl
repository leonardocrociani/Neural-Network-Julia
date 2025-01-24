include("../src/lib/Tensor.jl")
include("../src/lib/Metrics.jl")

using .Tensors
using .Metrics


function test_metrics()                 
    
    # Ok. testato anche con python
    # https://colab.research.google.com/drive/18T0rKt-uNPv7b6HDEnVlXl-mOsuI_ajU?usp=sharing
    
    println("=== Test 1 ===")
    y_pred_1 = Tensor([1.0 0.0 1.0 0.0])
    y_true_1 = [0.0 1.0 1.0 0.0]

    # 0.5 percento di accuratezza.
    # Recall: 0.5
    # Precision: 0.5
    # F1 Score: 0.5

    println("Accuracy: ", Metrics.accuracy(y_pred_1, y_true_1))
    println("Recall: ", Metrics.recall(y_pred_1, y_true_1))
    println("Precision: ", Metrics.precision(y_pred_1, y_true_1))
    println("F1 Score: ", Metrics.f1_score(y_pred_1, y_true_1))

    println("=== Test 2 ===")
    y_pred_2 = Tensor([0.0 1.0 1.0 1.0])
    y_true_2 = [0.0 1.0 0.0 1.0]
    
    # Accuracy: 0.75
    # Recall: 0.75
    # Precision: 0.75
    # F1 Score: 0.75
    
    println("Accuracy: ", Metrics.accuracy(y_pred_2, y_true_2))
    println("Recall: ", Metrics.recall(y_pred_2, y_true_2))
    println("Precision: ", Metrics.precision(y_pred_2, y_true_2))
    println("F1 Score: ", Metrics.f1_score(y_pred_2, y_true_2))
    
    
    println("=== Test 3 ===")
    y_pred_3 = Tensor([1.0 1.0 1.0 1.0])
    y_true_3 = [1.0 1.0 1.0 1.0]
    
    # Accuracy: 1.0
    # Recall: 1.0
    # Precision: 1.0
    # F1 Score: 1.0
    
    println("Accuracy: ", Metrics.accuracy(y_pred_3, y_true_3))
    println("Recall: ", Metrics.recall(y_pred_3, y_true_3))
    println("Precision: ", Metrics.precision(y_pred_3, y_true_3))
    println("F1 Score: ", Metrics.f1_score(y_pred_3, y_true_3))
    
    
    println("=== Test 4 ===")
    y_pred_4 = Tensor([0.0 0.0 0.0 0.0])
    y_true_4 = [1.0 1.0 1.0 1.0]
    
    # Accuracy: 0.0
    # Recall: 0.0
    # Precision: 0.0
    # F1 Score: 0.0
    
    println("Accuracy: ", Metrics.accuracy(y_pred_4, y_true_4))
    println("Recall: ", Metrics.recall(y_pred_4, y_true_4))
    println("Precision: ", Metrics.precision(y_pred_4, y_true_4))
    println("F1 Score: ", Metrics.f1_score(y_pred_4, y_true_4))
    
    
    println("=== Test 5 ===")
    y_pred_5 = Tensor([1.0 0.0 0.0 1.0])
    y_true_5 = [0.0 0.0 1.0 1.0]
    
    # Accuracy: 0.5
    # Recall: 0.5
    # Precision: 0.5
    # F1 Score: 0.5
    
    println("Accuracy: ", Metrics.accuracy(y_pred_5, y_true_5))
    println("Recall: ", Metrics.recall(y_pred_5, y_true_5))
    println("Precision: ", Metrics.precision(y_pred_5, y_true_5))
    println("F1 Score: ", Metrics.f1_score(y_pred_5, y_true_5))
    

end

test_metrics()