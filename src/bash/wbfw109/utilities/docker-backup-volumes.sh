#!/bin/bash
function help()
{
  echo "🔎 Usage: docker-backup-volumes.sh <volume_name_1> [<volume_name_2>, ...]"
  echo "'docker_backup_volumes.sh': backup your docker volumes into 'docker/volume_backup/' dir."
  echo "  - arugments: docker volume name"
}

while [ -n "$1" ]; do
  case "$1" in
    -h | --help )
      help
      exit ;;
    -- )
      exit ;;
    * )
      echo -e "❔ unrecognized option '$1'\n"
      help
      exit ;;
  esac
  shift
done


if [[ -z $1 ]]; then
  help
  exit 1
fi

backup_dir="docker/volume_backup"
eval mkdir --parents $backup_dir
declare -a input_volume_name_array=($@)
declare -a volume_name_array=($(docker volume ls --format "{{.Name}}"))

for input_volume_name in ${input_volume_name_array[@]}; do
  if [[ " ${volume_name_array[@]} " =~ " ${input_volume_name} " ]]; then
    eval docker run -v "$input_volume_name":/dbdata --name dbstore ubuntu /bin/bash
    current_date_time=$(date --iso-8601=seconds)
    eval docker run --rm --volumes-from dbstore -v $(pwd):/backup ubuntu tar cvf "/backup/$backup_dir/$input_volume_name-backup-${current_date_time//:/}".tar /dbdata
    eval docker container rm dbstore
    echo "✅ [INFO] Backup complete the volume: $input_volume_name"
  else
    echo "❓ [Unknown] Pass volume which not exists: $input_volume_name"
  fi
done

