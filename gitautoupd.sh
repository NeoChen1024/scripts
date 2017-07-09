#!/bin/bash
dir="${1:-.}"
cd $dir
git add --all
git commit -m "${2:-`uuidgen`}"
git push -u origin master
