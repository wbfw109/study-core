#!/bin/bash
function shift_command() (
  rurl="$1" localdir="$2" && shift 2
  echo "$1"
  echo "??"
  echo "$localdir"
  echo "??"
  echo "$*"
)

shift_command "a" "b" "c"
