# deprecated
#!/bin/bash
help()
{
  echo "[INFO] Command Helper: Upload public repository 'study-python-core' from this repository."
  echo "  - require only one arugment: git commit message"
}

if [[ -z $1 ]]
then
  help
  exit 1
elif [[ -n $2 ]]
then
  help
  exit 1
fi


temp_archive_target="$(pwd)/../study-core-python_temp.tar"
target_uznip_dir="$(pwd)/../_copy_temp/study-python-core/"

git archive main --format=tar --output=$temp_archive_target
cd $target_uznip_dir
rm -r $(ls --almost-all | grep --extended-regexp --invert-match '^\.git$')
tar --extract --file=$temp_archive_target
rm $temp_archive_target

git add . && git commit -m "$1" && git push origin main

# bash src/bash/wbfw109/utilities/self/archive_and_unzip_to_copy.sh "message"