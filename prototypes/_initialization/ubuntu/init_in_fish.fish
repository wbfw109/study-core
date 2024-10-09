#!/usr/bin/env fish
# Define the Fish configuration file path. Ensure Path.
set -Ux FISH_CONFIG "$HOME/.config/fish/config.fish"
set -Ux FISH_COMPLETIONS "$HOME/.config/fish/completions"
set BASHRC_PATH "$HOME/.bashrc"

### Update and upgrade packages
sudo apt -y update

# 📝 Note that this must be precedded than other setting because it adds "set PATH $PATH /home/wbfw109v2/.local/bin".
##### pipx settings 
sudo apt install -y pipx
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

# Append the necessary commands to the Fish config file if not already present
if not grep --quiet "### pyenv settings" "$FISH_CONFIG"
    echo "" >> "$FISH_CONFIG"
    echo "" >> "$FISH_CONFIG"
    echo "### pyenv settings" >> "$FISH_CONFIG"
    echo "pyenv init - | source" >> "$FISH_CONFIG"
end

echo "Commands added to $FISH_CONFIG"

# Check if ~/.bashrc already contains the Fish shell redirect
set FISH_REDIRECT_COMMAND 'if [ -n "$SSH_CONNECTION" ]; then\n    exec fish\nfi'

if not grep --quiet "exec fish" "$BASHRC_PATH"
    # Add the Fish shell redirect to ~/.bashrc if not already present
    echo -e "\n$FISH_REDIRECT_COMMAND" >> "$BASHRC_PATH"
end



# Define the content to be added to the Fish config file.
set config_content "
# Check and Set DISPLAY for SSH Connections and Tailscale 📅 Last updated date: 2024-10-10 01:35:47
if set --query SSH_CONNECTION
    set client_ip (echo \$SSH_CONNECTION | awk '{print \$1}')
    
    # If the SSH client IP is 127.0.0.1, it may be due to tunneling through localhost,
    # such as when using VSCode Remote-SSH, a proxy jump, or other SSH forwarding setups.
    # In this case, we check if Tailscale is available as one possible solution to find the actual client IP.
    if test \"\$client_ip\" = \"127.0.0.1\"
        # Check if Tailscale is installed
        if test (which tailscale) != \"\"
            # Get the client IP from Tailscale if there's an active direct connection
            set client_ip (tailscale status | grep \"active; direct\" | awk '{print \$1}')
            
            # If a client IP was found, set the DISPLAY variable
            if test -n \"\$client_ip\"
                set --export DISPLAY \"\$client_ip:0\"
            else
                echo \"No active Tailscale client IP found.\"
            end
        else
            echo \"Tailscale is not installed.\"
        end
    else
        # Set DISPLAY variable if SSH_CONNECTION is not localhost
        set --export DISPLAY \"\$client_ip:0\"
    end
end
"

# Check if the configuration already exists to avoid duplication.
if not grep -q "# Check and Set DISPLAY for SSH Connections and Tailscale" $FISH_CONFIG
    # Append the configuration to the Fish config file using tee.
    echo $config_content | tee -a $FISH_CONFIG > /dev/null
    echo "Configuration added successfully to $FISH_CONFIG"
else
    echo "Configuration already exists in $FISH_CONFIG"
end



##### Packages Managed by "apt" command in "fish" shell
pipx install poetry conan


#### Install Microsoft Edge
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/repos/edge stable main" | sudo tee /etc/apt/sources.list.d/microsoft-edge.list > /dev/null
sudo apt update -y; and sudo apt install microsoft-edge-stable




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

