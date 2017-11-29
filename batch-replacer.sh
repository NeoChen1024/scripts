#!/bin/bash
# A script for replacing strings in a lot of files
# Use with xargs and find
# Example: $ find ./text -print0 | xargs -0 batch-replacer -r "Error" -t "Correct"

set -e
all="$#"
printf "\e[s"
declare -i count=0
SCRIPT_NAME="batch-replacer"
AURTHOR="Neo_Chen <chenkolei@gmail.com>"
LICENSE="GPLv3 or Newer"

# usage() is important
usage()
{
echo \
"Usage: $SCRIPT_NAME [-v] [-r <string>] [-a <string>] [-h]
	-v		Be verbose, one more -v gives more details
	-r <string1>	Set replace target to <string1>
	-t <string2>	Replace <string1> with <string2>
	-h		This message

${AURTHOR}, licensed under ${LICENSE}"
}

# Get command line aruguments

while getopts "hf:v" OPTION ; do
	case $OPTION in
	h)	usage && exit 1 ;;
	v)	VERBOSE=1 ;;
	r)	REPLACE="$OPTARG" ;;
	t)	TO="$OPTARG" ;;
	esac
done

for i in "$@" ; do
	count=$((++count))
	printf "[%s/%s]\e[1;34m>>\e[32m%s\e[1;34m<<\e[0m" "$count" "$all" "$i"
	if grep "$REPLACE" "$i" > /dev/null 2>&1 ; then
		sed -i "s|${REPLACE}|${TO}|" "$i"
		printf "\n\t"
		printf "\e[2K\e[u"
	fi
done
exit 0
