#!/usr/bin/env fish
# Define the Fish configuration file path. Ensure Path.
set -Ux FISH_CONFIG "$HOME/.config/fish/config.fish"
set -Ux FISH_COMPLETIONS "$HOME/.config/fish/completions"


### update and upgrade packages
sudo apt update -y


📝 Note that this must be precedded than other setting because it adds "set PATH $PATH /home/wbfw109v2/.local/bin".
##### pipx settings 
sudo apt install pipx
pipx ensurepath
#💡 'set PATH $PATH ~/.local/bin' is automatically registered in $FISH_CONFIG" by pipx

# Not required to be in the config file, only run once
register-python-argcomplete --shell fish pipx >$FISH_COMPLETIONS/pipx.fish



##### Packages Managed by "apt" command in "fish" shell
### pyenv settings
curl https://pyenv.run | bash

# Set PYENV_ROOT environment variable
set -Ux PYENV_ROOT $HOME/.pyenv
fish_add_path $PYENV_ROOT/bin

# Append the necessary commands to the Fish config file

echo "" >> "$FISH_CONFIG"
echo "" >> "$FISH_CONFIG"
echo "### pyenv settings" >> "$FISH_CONFIG"
echo "pyenv init - | source" >> "$FISH_CONFIG"

echo "Commands added to $FISH_CONFIG"





##### Packages Managed by "apt" command in "fish" shell
pipx install poetry conan



##### Custom commands
echo "" >> "$FISH_CONFIG"
echo "" >> "$FISH_CONFIG"
echo "# For integration with VSCode's 'python.defaultInterpreterPath' key-value" >> "$FISH_CONFIG"
echo "set PYTHON_POETRY_BASE_EXECUTABLE (which python3)" >> "$FISH_CONFIG"




# #### Privacy setting .. TODO: make as function
# git config --global init.defaultBranch main
# git config --global user.name "wbfw109v2"
# git config --global user.email "wbfw109v2@gmail.com"
# git remote add origin https://github.com/abcde111112/intel-edge-academy-6.git

