#!/bin/bash
function help()
{
  echo "🔎 Usage: merge_with_rebase_and_reinit_dev.sh (-m | --message) <message>"
  echo "'merge_with_rebase_and_reinit_dev': [Git] merge dev to main with rebase and re-initialize dev branch"
  echo "Mandatory Options"
  echo "  -m, --message        squashed commit message to main branch"
}


commit_message="."
while [ -n "$1" ]; do
  case "$1" in
    -m | --message )
      shift; commit_message=$1
      ;;
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
dev_branch="dev"
main_branch="main"

eval "git checkout $dev_branch && git rebase $main_branch" &&
eval "git checkout $main_branch && git merge --squash $dev_branch && git commit -m '$commit_message' && git push origin $main_branch" &&
eval "git branch -D $dev_branch && git checkout -b $dev_branch"
