#!/bin/bash
# Git-Autopull script -- update all repositorys under a directory
# Usage : git-autopull.sh [dir]
set -ev
if [ -v $1 ]||[ $1 == "." ]; then
	olddir="$PWD"
	cd "$1"
fi

for i in */ ;do
	cd "$i"
	echo -e "\e[1;34m>>\e[32m$i\e[1;34m<<\e[0m"
	git pull
	echo -e "\e[36m========\e[0m\n"
	cd ..
done
echo "Done!"
cd "$olddir"
return 0
