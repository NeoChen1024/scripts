#!/bin/bash
# Git-Autopull script -- update all repositorys under a directory
# Usage : git-autopull.sh [dir]
set -e
(
if [ -v "$1" ]||[ "$1" == "." ]; then
	cd "$1"
fi

for i in */ ;do
	(
	cd "$i"
	printf '\e[1;34m>>\e[32m%s\e[1;34m<<\e[0m' "$i"
	git pull
	printf '\e[36m========\e[0m\n'
	)
done
)
echo "Done!"
exit 0
