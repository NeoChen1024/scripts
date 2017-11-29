#!/bin/bash
# Bash Script Template
[ -x "$HOME/.local/lib/neostdlib.sh" ] && source "$HOME/.local/lib/neostdlib.sh"
[ -x "/usr/lib/neostdlib.sh" ] && source "/usr/lib/neostdlib.sh"

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

## Colors
ESC="\033"
RESET="${ESC}[0m"	#Reset all attributes
BRIGHT="${ESC}[1m"	#Bright
DIM="${ESC}[2m"	#Dim
BLINK="${ESC}[5m"	#Blink
#Foreground Colours
FBLACK="${ESC}[30m"	#Black
FRED="${ESC}[31m"	#Red
FGREEN="${ESC}[32m"	#Green
FYELLOW="${ESC}[33m"	#Yellow
FBLUE="${ESC}[34m"	#Blue
FMAGENTA="${ESC}[35m"	#Magenta
FCYAN="${ESC}[36m"	#Cyan
FWHITE="${ESC}[37m"	#White
#Background Colours
BBLACK="${ESC}[40m"	#Black
BRED="${ESC}[41m"	#Red
BGREEN="${ESC}[42m"	#Green
BYELLOW="${ESC}[43m"	#Yellow
BBLUE="${ESC}[44m"	#Blue
BMAGENTA="${ESC}[45m"	#Magenta
BCYAN="${ESC}[46m"	#Cyan
BWHITE="${ESC}[47m"	#White

echo -e "${FGREEN}This ${FYELLOW}script ${FCYAN}does ${FMAGENTA}nothing.${RESET}"
