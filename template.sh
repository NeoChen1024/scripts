#!/usr/bin/env sh
# Bash Script Template

# %DYN_LOAD%
# Search for "neostdlib.sh" (Neo_Chen's Shell Script Standard Library)
#for i in "/usr/lib" "/usr/local/lib" "/lib" "$HOME/lib" "$HOME/.local/lib"
#do
#	if [ -x "$i/neostdlib.sh" ] ;then
#		source "$i/neostdlib.sh"
#		break
#	fi
#done
# %END_DYN_LOAD%

SCRIPT_NAME="Template"
AURTHOR="Name <email addr>"
LICENSE="Public Domain"

# usage() is important
usage()
{
	echo "Usage: $SCRIPT_NAME [-v] [-h] [-f filename]"
}

# Get command line aruguments

while getopts "hf:v" OPTION ; do
	case $OPTION in
	h)
		usage
		exit 1
		;;
	f)
		FILE=$OPTARG
		;;
	v)
		VERBOSE=1
		;;
	esac
done

echo "This script does nothing."
# vim: set tabstop=8:softtabstop=8:shiftwidth=8
