#!/bin/bash
# Bash Script Template
SCRIPT_NAME="Template"
AURTHOR="Name <email addr>"
LICENSE="GPLv3 or Newer"

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
