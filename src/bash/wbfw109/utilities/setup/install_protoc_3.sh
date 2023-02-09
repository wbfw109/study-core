#!/bin/bash
function help()
{
  echo "🔎 Usage: install_protoc_3.sh [-u | --update]"
  echo "'install_protoc_3.sh': install Protocolf Buffers Compiler 3 as linux-x86_64 distribution from https://github.com/protocolbuffers/protobuf"
  echo "❕ sudo required to move files into /usr/local/include and /usr/local/bin".
  echo -e "  - used third-party libraries: jq, semver\n"
  echo "Options"
  echo "  -u, --update         if protobuf compiler (protoc) already installed, update."
  echo "                       it uses major versioning of semver."
}

UPDATE=false
while [ -n "$1" ]; do
  case "$1" in
    -u | --update )
      UPDATE=true ;;
    -h | --help )
      help
      exit ;;
    -- )
      break ;;
    * )
      echo -e "❔ unrecognized option '$1'\n"
      help
      exit ;;
  esac
  shift
done

if [[ "$EUID" -ne 0 ]]
  then echo "❗ Please run as root"
  help
  exit
fi

# check existing protoc version
if [[ -n "$(which protoc)" ]]; then
  declare -a protoc_version_info=($(protoc --version))
  protoc_version=($(echo ${protoc_version_info[@]:1:2}))
  max_version=$(eval semver bump major $protoc_version)
else
  protoc_version=-1
  max_version=-1
fi

# check required binaries. key: binary file name, value: installation command
declare -A required_binaries=(["jq"]="'sudo apt install jq'" ["semver"]="'bash src/bash/wbfw109/utilities/setup/install_protoc_3.sh'")
declare -A error_messages=()
should_break=false
for binary in ${!required_binaries[@]}; do
  if [[ -z "$(which $binary)" ]]; then
    error_messages[${#error_messages[@]}]="  - $binary. run ${required_binaries["$binary"]}"
    should_break=true
  fi
done
if [[ "$should_break" = true ]]; then
  echo "❓ Can not start install. firstly install [Required packages]:"
  printf "%s\n" "${error_messages[@]}"
  exit
fi


api_result=$(curl \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/protocolbuffers/protobuf/releases)

tag_names=$(jq '[.[].tag_name]' <<< $api_result)
# versioning of release was changed from major version 21. (v3.20.3 -> v21.1)
# filter prerelase version. e.g. (X): v21.0-rc1   (O): v21.12
valid_versions=$(jq 'map(select(startswith("v3.") | not))
  | map(select(contains("-") | not))' <<< $tag_names)
# substitude strings. e.g. "v21.0" to "3.21.0"
valid_semvers=$(jq 'map(("3." + .[1:]))' <<< $valid_versions)

target_valid_semver="0.0.0"
target_index=0
index=0
for valid_semver in $(jq --raw-output '.[]' <<< $valid_semvers); do
  # result value of jq remains quotes, so to use as arguments in semver command those quotes must be stripped.
  if [[ "$(semver compare $target_valid_semver $valid_semver)" -lt 0 ]]; then
    target_valid_semver=$valid_semver
    target_index=$index
  fi
  index=($index + 1)
done
target_valid_version=$(jq --argjson target_index $target_index '.[$target_index]' <<< $valid_versions)
target_release_assets=$(jq --argjson target_valid_version $target_valid_version '.[]
  | select(.tag_name==$target_valid_version).assets' <<< $api_result)

function install_protoc()
{
  echo 📦 install Protocol Compiler
  temp_path="/tmp"
  protoc_target_release_stem="protoc-${target_valid_version:2:-1}-linux-x86_64"
  protoc_zip_filename="$protoc_target_release_stem.zip"
  target_asset=$(jq '.[] | select(.name=="'"$protoc_zip_filename"'")' <<< $target_release_assets)
  browser_download_url=$(jq --raw-output '.browser_download_url' <<< $target_asset)

  protoc_zip_path="$temp_path/$protoc_zip_filename"
  protoc_output_path="$temp_path/$protoc_target_release_stem"
  protobuf_common_types_dir="/usr/local/include/google/protobuf"


  # overwrite to /tmp/
  eval curl --location --show-error $browser_download_url > $protoc_zip_path &&
  eval unzip -o $protoc_zip_path -d $protoc_output_path &&
  eval sudo rsync -a $protoc_output_path/bin/ /usr/local/bin/ &&
  if [[ $UPDATE = true && -d $protobuf_common_types_dir ]]; then
    rm -rf $protobuf_common_types_dir
  fi &&
  eval sudo rsync -a $protoc_output_path/include/ /usr/local/include/ &&
  echo "✅ [INFO] complete protoc installation: version $(protoc --version)"
}

# if already protoc was installed,
if [[ $protoc_version != -1 ]]; then
  if [[ $protoc_version != $target_valid_semver ]]; then
    echo "Your protoc version is $protoc_version. New version available: $target_valid_semver ($target_valid_version)"
    echo "Please --update | -u option for update existing protoc."
    if [[ $UPDATE = true ]]; then
      if [[ "$(semver compare $target_valid_semver $max_version)" -lt 0 ]]; then
        install_protoc
      else
        echo "❕ Pass protoc update because major semver of $protoc_version (~ $max_version (exclusive)) not meets $target_valid_semver"
      fi
    fi
  else
    echo "Your protoc version up to date."
  fi
else
  install_protoc
fi
