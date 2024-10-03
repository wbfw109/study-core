# Set the base path to the MSVC tools directory
$basePath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"

# Find the latest version directory
$latestVersionPath = Get-ChildItem -Path $basePath -Directory | 
    Sort-Object Name -Descending | 
    Select-Object -First 1 | 
    ForEach-Object { $_.FullName }

# Set the target path to the 'bin\Hostx64\x64' directory of the latest version
$targetPath = Join-Path -Path $latestVersionPath -ChildPath "bin\Hostx64\x64"

# Get the current PATH environment variable
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::Machine)

# Define the pattern to remove
# Use a pattern to match directories similar to the '14.*\bin\Hostx64\x64'
$pattern = [regex]::Escape($basePath) + "\\[^;]*?\\bin\\Hostx64\\x64"

# Remove old paths matching the pattern from the current PATH
$newPath = $currentPath -replace "([^;]*$pattern[^;]*;?)", ""

# Append the new path to the updated PATH
$newPathValue = "$newPath;$targetPath"
[System.Environment]::SetEnvironmentVariable("PATH", $newPathValue, [System.EnvironmentVariableTarget]::Machine)

Write-Output "Updated PATH successfully. New path added: $targetPath"


### debugging: Get-Variable | Where-Object { $_.Name -like "*path*" }  