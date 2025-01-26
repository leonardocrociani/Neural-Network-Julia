module DataLoader

    using DelimitedFiles

    export load_monks_data

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
end