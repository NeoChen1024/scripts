#!/bin/bash
dir="${2:-.}"
cd $dir
git add --all
git commit -m "${1:-`uuidgen`}"
git push -u origin master
