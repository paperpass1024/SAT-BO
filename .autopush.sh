#!/bin/bash
SCRIPT_DIR="$(readlink -f "$( dirname "${BASH_SOURCE[0]}" )" )"
SCRIPT_DIR_NAME=$(basename "$SCRIPT_DIR" )
cd "$SCRIPT_DIR" || return
git add .
git commit -m $today
git push "$SCRIPT_DIR_NAME" master 
