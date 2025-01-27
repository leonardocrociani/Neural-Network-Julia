module DataLoader

    using DelimitedFiles

    export load_monks_data, load_cup_data   

    function load_monks_data(train_files::Vector{String}, test_files::Vector{String})

        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        for i in eachindex(train_files)
            train_file = train_files[i]
            test_file = test_files[i]

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

        end

        println("Loading data...")

        return X_train, Y_train, X_test, Y_test
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

end