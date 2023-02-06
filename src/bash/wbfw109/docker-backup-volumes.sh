#!/bin/bash
help()
{
  echo "[INFO] Command Helper: backup your docker volumes into 'docker/volume_backup/' dir."
  echo "  - arugments: docker volume name"
}

if [[ -z $1 ]]
then
  help
  exit 1
fi

backup_dir="docker/volume_backup"
eval mkdir --parents $backup_dir
declare -a input_volume_name_array=($@)
volume_name_str=$(docker volume ls --format "{{.Name}}")
declare -a volume_name_array=($volume_name_str)

for input_volume_name in ${input_volume_name_array[@]}
do
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

