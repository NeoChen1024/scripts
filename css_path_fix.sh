#!/bin/bash
set -e
TMPFILE="/tmp/sed-$UID.tmp"
DIR=${1:-$HOME/www}
printf "\e[s"
all=$(find "$DIR" -name \*.html | wc -l)
declare -i count=0

for i in $(find "$DIR" -name \*.html) ; do
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
