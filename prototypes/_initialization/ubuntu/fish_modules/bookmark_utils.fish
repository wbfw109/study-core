#!/usr/bin/env fish
# Written at 📅 2024-10-28 13:48:11
: '
🛍️ Usage e.g.
# Load my modules
set -x FISH_MODULES_PATH prototypes/_initialization/ubuntu/fish_modules

source $FISH_MODULES_PATH/bookmark_utils.fish

add_bookmark $LOCAL_MOUNT_PATH
echo "Bookmark update completed!"
'

# bookmark_utils.fish

# Set the default path for Ubuntu bookmarks
# gtk: GIMP(GNU Image Manipulation Program) Toolkit
set -g UBUNTU_BOOKMARKS_PATH "$HOME/.config/gtk-3.0/bookmarks"


# Function to print usage information and exit
function print_usage_and_exit
    echo "Usage: add_bookmark <path_to_add> [optional_bookmarks_path]"
    exit 1
end

# Function to check if the provided path is valid
function validate_path
    set path_to_check $argv[1]

    # (short: -e) - Checks if the path (file or directory) exists.
    if not test -e $path_to_check
        echo "Error: Invalid path '$path_to_check'. Please provide a valid path."
        exit 1
    end
end

# Function to add a bookmark if it doesn't already exist
function add_bookmark
    # Check if the required argument is provided
    if test (count $argv) -lt 1
        echo "Error: Missing required argument <path_to_add>"
        print_usage_and_exit
    end

    # Set the path to add and the bookmarks path (default if not provided)
    set path_to_add $argv[1]
    set bookmarks_path (if test (count $argv) -gt 1; echo $argv[2]; else; echo $UBUNTU_BOOKMARKS_PATH; end)

    # Validate if the provided path is valid
    validate_path $path_to_add

    # Convert local path to file URI format
    set file_uri (string replace -r '^/' 'file:///' $path_to_add)

    # (short: -F, long: --fixed-strings) - Treat the pattern as a literal string, not a regular expression.
    # (short: -x, long: --line-regexp) - Ensure the whole line matches the pattern exactly.
    # (short: -q, long: --quiet) - Suppress output; only the exit status is used to indicate a match.
    if not grep -Fxq $file_uri $bookmarks_path
        echo "Adding: $file_uri"
        echo $file_uri >> $bookmarks_path
    else
        echo "Bookmark already exists: $file_uri"
    end
end


