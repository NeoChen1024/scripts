#!/bin/bash
# A script for fixing my CSS Path Problem
# Use with xargs and find
# Example: $ find ./www -name \*.html -print0 | xargs -0 ./css_path_fix.sh

set -e
TMPFILE="/tmp/$RANDOM"
all="$#"
printf "\e[s"
declare -i count=0
for i in "$@" ; do
	count=$((++count))
	printf "[%s/%s]\e[1;34m>>\e[32m%s\e[1;34m<<\e[0m" "$count" "$all" "$i"
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
