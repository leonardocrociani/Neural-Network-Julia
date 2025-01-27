module DataLoader

	using DelimitedFiles

	export load_monks_data, load_cup_data, save_cup_results

	function onehot_encode(y_true::Vector{Int64}, num_classes::Int)
		# Initialize a matrix of zeros with size (num_samples, num_classes)
		num_samples = length(y_true)
		y_onehot = zeros(Int, num_samples, num_classes)

		# Set the appropriate indices to 1
		for (i, class_idx) in enumerate(y_true)
				y_onehot[i, class_idx + 1] = 1  # +1 because Julia uses 1-based indexing
		end

		return y_onehot
	end

	function load_monks_data(train_file::String, test_file::String)

		X_train = []
		Y_train = Int[]

		X_test = []
		Y_test = Int[]

		# training
		# read line by line
		open(train_file) do file
			for line in eachline(file)
				line = strip(line)
				line = split(line, " ")

				if length(line) <= 1
					continue
				end

				pop!(line) # last is the identifier
				
				x = parse.(Int, line[2:end])
				y = parse(Int, line[1])
				push!(X_train, x)
				push!(Y_train, y)
			end
		end

		open(test_file) do file
			for line in eachline(file)
				line = strip(line)
				line = split(line, " ")

				if length(line) <= 1
					continue
				end

				pop!(line) # last is the identifier
				
				x = parse.(Int, line[2:end])
				y = parse(Int, line[1])
				push!(X_test, x)
				push!(Y_test, y)
			end
		end

		println("Loading data...")

		return X_train, onehot_encode(Y_train, 2), X_test, onehot_encode(Y_test, 2)
	end


	function load_cup_data(filename::String; test_set::Bool=false) 
		X = []
		Y = []

		first_line = false

		open(filename) do file
			for line in eachline(file)

				if !first_line
					first_line = true
					continue
				end

				line = strip(line)
				line = split(line, ",")

				if length(line) <= 1
					continue
				end

				if test_set
					x = parse.(Float64, line[2:end])
					push!(X, x)
					continue
				end

				# last 3 columns are the target
				x = parse.(Float64, line[2:end-3])
				y = parse.(Float64, line[end-2:end])

				push!(X, x)
				push!(Y, y)
			end
		end

		if test_set
			return X
		end

		return X, Y
	end


	function save_cup_results(Y_pred::Array{Array{Float64, 1}, 1})
		@assert length(Y_pred) == 500
		@assert all([length(y) == 3 for y in Y_pred])

		ln1 = "# Giovanni Braccini, Leonardo Crociani, Giacomo Trapani\n"
		ln1 = ln1 * "# JTeam\n" 
		ln1 = ln1 * "# ML-CUP24 V1\n"
		ln1 = ln1 * "# 29/01/2025\n"

		for i in 1:length(Y_pred)
			y = Y_pred[i]
			ln1 = ln1 * string(i) * "," * string(y[1]) * "," * string(y[2]) * "," * string(y[3]) * "\n"
		end

		open("../outputs/ML_CUP_24_TS_results.csv", "w") do file
			write(file, ln1)
		end
	end

end