include("lib/DataLoader.jl")

using .DataLoader

file_train = "../dataset/cup/ML-CUP24-TR.csv"
file_test = "../dataset/cup/ML-CUP24-TS.csv"

X_train, Y_train = load_cup_data(file_train)

X_test = load_cup_data(file_test, test_set=true) # only for test (no valudation since no label)

println("X_train: ", size(X_train))
println("Y_train: ", size(Y_train))
println("X_test: ", size(X_test))

# once you have the outputs of the test, just run the following code to save the results in the right format
Y_pred = [rand(3) for _ in 1:length(X_test)]
save_cup_results(Y_pred)