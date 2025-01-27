include("lib/DataLoader.jl")

using .DataLoader

file_train = "../dataset/cup/ML-CUP24-TR.csv"

X_train, Y_train = load_cup_data(file_train)