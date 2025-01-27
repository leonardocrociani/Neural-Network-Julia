module DataLoader

	using DelimitedFiles

	export load_monks_data, load_cup_data, save_cup_results

	function encode_monk(data::Vector{Int64})
		max_values = [3,3,2,3,4,2]
		encoded_inputs = []
		for i in eachindex(data)
			for j in range(1, max_values[i])
				if j + 1 == data[i]
					push!(encoded_inputs, 1)
				else
					push!(encoded_inputs, 0)
				end
			end
		end
		return encoded_inputs
	end

	function load_monks_data(file_path::String)
		dataset = []
		open(file_path) do file
			for line in eachline(file)
				words_list = split(strip(line), " ")
				if length(words_list) <= 1
					continue
				end
				id = words_list[8]
				output_class = parse(Int64, words_list[1])
				inputs = parse.(Int64, words_list[2 : 7])
				inputs = encode_monk(inputs)
				push!(dataset, (id, inputs, output_class))
			end
		end

		return dataset
	end


	function load_cup_data(filename::String; test_set::Bool=false) 
		X = Float64[]
		Y = Float64[]

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