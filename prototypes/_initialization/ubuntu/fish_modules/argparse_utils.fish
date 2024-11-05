#!/usr/bin/env fish
# Written at 📅 2024-10-28 13:48:11

function check_argv_or_usage_and_exit
    : '
    🚧 Prerequisites
      define "usage" function
      pass argv[1] to function

    Usage 🛍️ e.g. 
        source $FISH_MODULES_PATH/argparse_utils.fish

        function usage
            echo Hello world
        end

        check_argv_or_usage_and_exit $argv[1]
        set main_command $argv[1]
        set --erase argv[1]
    '
    set -l arg $argv[1]
    if test -z "$arg"
        usage
        exit 1
    end
end
