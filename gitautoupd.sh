#!/bin/bash
dir="${2:-.}"
cd "$dir" || exit 1
git add --all
git commit -m "${1:-$(uuidgen)}"
git push -u origin master
