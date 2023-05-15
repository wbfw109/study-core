#!/usr/bin/bash
function usage() {
    echo "Usage: $0 [--dry-run]"
    echo "  --dry-run  : Perform a dry-run of git clean without deleting files."
}

## validate
if [[ $# -gt 1 ]]; then
    echo "Error: Too many arguments."
    usage
    exit 1
fi

# if first argument exists but is not --dry-run.
if [[ -n "$1" && "$1" != "--dry-run" ]]; then
    echo "Error: Invalid argument '$1'."
    usage
    exit 1
fi


## process
if [[ "$1" == "--dry-run" ]]; then
    echo "Performing dry-run with git clean..."
    git clean -fdxn
else
    echo "Cleaning up the project and creating tar archive..."
    git clean -fdx \
    && tar -cvf ~/cTest.tar ../intel-edge-academy-6
fi
