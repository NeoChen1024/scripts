#!/bin/bash
# A script for fixing my CSS Path Problem
# Use with xargs

set -e
TMPFILE="/tmp/$RANDOM"
DIR="$HOME/www"
all="$#"
printf "\e[s"
declare -i count=0

for i in "$@" ; do
	count=$((++count))
	printf "[$count/$all]\e[1;34m>>\e[32m$i\e[1;34m<<\e[0m"
	if grep 'href="/neostyle.css"' "$i" > /dev/null 2>&1 ; then
		sed 's|href="/neostyle.css"|href="/~neo_chen/neostyle.css"|' "$i" > "$TMPFILE"
		printf "\n\t"
		cp -v "$TMPFILE" "$i"
		printf "\e[s"
	else
		printf "\e[2K\e[u"
	fi
done
exit 0
