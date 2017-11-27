#!/bin/bash
# mirror-sync: a script make you can sync much file mirrors easily
# Copyright (C) 2017 陳珂磊 (Neo_Chen)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ==============================
# Document For Config File:
# ==============================
# The config file uses same syntax as shell script,
# declare -i SERVER_MAX_INDEX="<number of your servers>"
# declare -a SERVERS[]=("<names of your servers>")
# SERVER_<name>_CMDLINE=""
#
# Available variables:
# QUIET (int)		-q is set
# VERBOSE (int)		-v is set (more -v makes $VERBOSE bigger)
# DIR (string)		-d is set or using default value

set -xe
SCRIPT_NAME="mirror-sync"
AURTHOR="Neo_Chen <chenkolei@gmail.com>"
LICENSE="GPLv3 or Newer"

# usage() is important
usage()
{
	echo "Usage: $SCRIPT_NAME [-v] [-f <configure file>] [-d] [-q] [-o <variable>=<value>] [-h]"
	echo "			-v	Be verbose, one more -v gives more details"
	echo "			-f	Use alternative configfile instead of ~/.mirrorsync"
	echo "			-d	Sync with different directory from .mirrorsync"
	echo "			-q	Make this script really quiet"
	echo "			-o	Add your own variable"
	echo "			-h	Displays this message"
	echo "	Made by ${AURTHOR}, licensed under ${LICENSE}"
}

# Set default variables
declare -i VERBOSE=0
declare -i QUIET=0
declare CONFIG="${HOME}/.mirrorsync"
declare DIR="${HOME}/HTML/"
declare -i COUNT=0

# Get command line aruguments

while getopts "f:d:qvh" OPTION ; do
	case $OPTION in
	h)	usage && exit 1 ;;
	f)	CONFIG="$OPTARG" ;;
	v)	((++VERBOSE)) ;;
	d)	DIR="$OPTARG" ;;
	q)	VERBOSE=0
		QUIET=1 ;;
	esac
done

# ===============Colors=============== #
ESC="\033"
RESET="${ESC}[0m"	#Reset all attributes
BRIGHT="${ESC}[1m"	#Bright
DIM="${ESC}[2m"	#Dim
BLINK="${ESC}[5m"	#Blink
# Foreground Colours #
FBLACK="${ESC}[30m"	#Black
FRED="${ESC}[31m"	#Red
FGREEN="${ESC}[32m"	#Green
FYELLOW="${ESC}[33m"	#Yellow
FBLUE="${ESC}[34m"	#Blue
FMAGENTA="${ESC}[35m"	#Magenta
FCYAN="${ESC}[36m"	#Cyan
FWHITE="${ESC}[37m"	#White
# Background Colours #
BBLACK="${ESC}[40m"	#Black
BRED="${ESC}[41m"	#Red
BGREEN="${ESC}[42m"	#Green
BYELLOW="${ESC}[43m"	#Yellow
BBLUE="${ESC}[44m"	#Blue
BMAGENTA="${ESC}[45m"	#Magenta
BCYAN="${ESC}[46m"	#Cyan
BWHITE="${ESC}[47m"	#White

msg_echo()
{
	if [ "$QUIET" -lt 1 ] ;then
		echo -e "${BBLUE}>>${RESET} ${FCYAN}${1}${RESET}"
	fi
}

verbose2_echo()
{
	if [ "$VERBOSE" -ge 2 ] ;then
		echo -e "${BRIGHT}${BBLUE}>>${RESET} ${BRIGHT}${FGREEN}${1}${RESET}"
	fi
}

is_set()
{
	local name
	local value
	name="$1"
	value="$2"

	if test -z "${value}" ;then
	msg_echo "${name} is not set"
	return 2
	fi
}

expand_variables()
{	
	verbose2_echo "expand_variables: $*"
	declare -g exp_name
	declare -g exp_cmdline
	local VAR
	count="$1"
	verbose2_echo "$count"
	exp_name="${SERVERS[${count}]}"
	verbose2_echo "exp_name=$exp_name"
	VAR="SERVER_${exp_name}_CMDLINE"
	verbose2_echo "VAR=$VAR"
	exp_cmdline="${!VAR}"
	verbose2_echo "exp_cmdline=$exp_cmdline"

	is_set "\$SERVERS[${count}]" "$exp_name"
	is_set "\$SERVER_${exp_name}_CMDLINE" "$exp_cmdline"
}

run_server()
{
	verbose2_echo "run_server: $*"
	local -i count
	eval "$1"
}

# Check files are exist
test -d "$DIR"
test -f "$CONFIG"

# Read the config
source "$CONFIG"

while [ "$SERVER_MAX_INDEX" -ge "$COUNT" ] ; do
	expand_variables "$COUNT"
	msg_echo "#[${count}]\t\"${exp_name}\", with commandline \"$exp_cmdline\""
	run_server "$exp_cmdline"
	let ++COUNT
done
