function poetry_install_requirements
    # Check if exactly one argument (base directory) is provided
    if test (count $argv) -ne 1
        echo "Error: Exactly one argument required, specifying the base directory."
        echo "Usage: poetry_install_requirements /path/to/project"
        return 1
    end

    # Get the base directory from the first argument
    set base_dir $argv[1]

    # Check if the requirements.txt file exists in the provided directory
    if not test -f "$base_dir/requirements.txt"
        echo "Error: requirements.txt not found in $base_dir"
        echo "Make sure the base directory contains a requirements.txt file."
        return 1
    end

    # Variable to store all non-referenced dependencies
    set non_group_dependencies

    # Iterate over each line in requirements.txt
    for line in (cat "$base_dir/requirements.txt")
        # Check if the line starts with "-r" (indicating a reference to another file)
        if string match -- "-r*" $line
            # Use echo and awk to extract the file path after "-r"
            set req_file (echo $line | awk '{print $2}')

            # Convert the relative path to an absolute path
            set req_file (realpath "$base_dir/$req_file")

            # Determine the group name based on the file name (without extension)
            set group_name (string split "/" $req_file)[-1]
            set group_name (string split "." $group_name)[1]

            # Read the dependencies from the referenced file
            set dependencies ""
            for dependency in (cat $req_file)
                # Remove unnecessary spaces in the dependency line
                set dependency (string trim -- $dependency)

                # Replace all occurrences of multiple spaces with no space
                while string match -q '* *' $dependency
                    set dependency (string replace -a ' ' '' $dependency)
                end

                # Add the cleaned dependency to the dependencies variable
                set dependencies $dependencies $dependency
            end

            # If there are dependencies, add them to the specified group using Poetry
            if test -n "$dependencies"
                echo "Running: poetry add --group $group_name $dependencies"
                eval "poetry add --group $group_name $dependencies"
            end
        else
            # If the line does not start with "-r", it's a direct dependency
            # Remove unnecessary spaces in the dependency line
            set line (string trim -- $line)

            # Replace all occurrences of multiple spaces with no space
            while string match -q '* *' $line
                set line (string replace -a ' ' '' $line)
            end

            # Add the cleaned dependency to non_group_dependencies
            set non_group_dependencies $non_group_dependencies $line
        end
    end

    # If there are non-group dependencies, add them all at once
    if test -n "$non_group_dependencies"
        echo "Running: poetry add $non_group_dependencies"
        eval "poetry add $non_group_dependencies"
    end
end

# Run the function with the provided argument
poetry_install_requirements $argv
