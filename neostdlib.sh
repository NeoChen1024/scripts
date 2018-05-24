#!/bin/bash
# vim: set tabstop=8:softtabstop=8:shiftwidth=8
########################################################################
# Standard Library For Neo_Chen's BASH Scripts
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
########################################################################

# Install it to /usr/lib/neostdlib.sh or $HOME/.local/lib/neostdlib.sh

# Declare some (un)common variable
declare SCRIPT_NAME
declare AURTHOR
declare LICENSE
declare -i VERBOSE
declare -i QUIET
declare CONFIG="${HOME}/.neoscriptrc"

	##########################################
	# =============== Colors =============== #
	##########################################

ESC='\\033'
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

	#########################
	# Functions:		#
	# Make your life easier #
	#########################

# Error Message that stops the script
error_echo()
{
# External Requirement: coreutils
# Arguent::	$* [any text]
	echo -e "${BRED}>>${RESET} ${FMAGENTA}${*}${RESET}"
	return 1
}

# Normal Message
msg_echo()
{
# External Requirement: coreutils
# Argument::	$* [any text]
	if [ "$QUIET" -lt 1 ] ;then
		echo -e "${BBLUE}>>${RESET} ${FCYAN}${*}${RESET}"
	fi
}

# Debug Level Verbose
debug_echo()
{
# External Requirement: coreutils
# Argument::	$* [any text]
	if [ "$VERBOSE" -ge 2 ] ;then
		echo -e "${BRIGHT}${BBLUE}>>${RESET} ${BRIGHT}${FGREEN}${*}${RESET}"
	fi
}

# Verbose Message
verbose_echo()
{
# External Requirement: coreutils
# Argument::	$* [any text]
	if [ "$VERBOSE" -ge 1 ] ;then
		echo -e "${BRIGHT}${BBLUE}>>${RESET} ${BRIGHT}${FCYAN}${*}${RESET}"
	fi
}

# Check whether the variable is set or not
is_set()
{
# Argument::	$name [variable name]
# Return Err::	When $name is not set or its length is zero
	local name
	name="$1"
	
	if [ -z "${!name}" ] ;then
		msg_echo "${name} is not set"
		return 2
	fi
}

# Check whether the file is exist or not
is_exist()
{
# Argument::	$file
# Return Err::	When $file is not exist
	local file
	file="$1"

	[ -f "$file" ] || return 2
}

# Expand variables from argument
expand_variables()
{
# Argument :: 	$<prefix>, $<class>, $<name>, $var_name_to
# Return Var ::	$<var_name_to>=$<prefix>_<class>_<name>
	local var_class_name
	local var_class_item
	local prefix
	local class

	prefix="$1"
	class="$2"
	name="$3"
	var_name="$4"

	var_class_name="${prefix}_${class}_${name}"
	declare "${var_name}=${!var_class_item}"

	is_set "$var_class_name" "${!var_class_name}"
}

# Generate random bytes
random()
{
# External Requirement: coreutils
# Argument :: $source (random|pseudo), $length [bytes]
	local source
	local length

	source="$1"
	length="$2"

	case "$source" in
		random)	dd if=/dev/random bs=1 count="$length" ;;
		pseudo)	dd if=/dev/urandom bs=1 count="$length" ;;
	esac
}

# Convert binary stream to hexdecimal
hexify()
{
# External Requirement: coreutils
# Argument :: $source (stdio|file)
	local source
	local file
	source="$1"
	file="$2"

	if [ "$source" == "stdio" ] ;then
		od -t x2
	elif [ "$source" == "file" ]&&[ -f "$file" ] ;then
		od -t x2 "$file"
	fi
}

# Fetch URL -- Non-tested
fetch_url()
{
# External Requirement: curl, netcat
# Argument :: $flags $url $directory_or_filename
# ---[Flags]----------------------------------------
#		<flag> -- <description>
#		n -- nothing, be normal
#		o -- do overwrite
#		- -- to stdout
#		t -- download to temporary directory
# --------------------------------------------------
# URL scheme can be : https, http, ftp, tftp, raw 
	local flag
	local url
	local fd
	local filename
	flag="$1"
	url="$2"
	fd="$3"

	case "$flag" in
		*o*) local flag_o=1 ;;
		*-*) local flag_dash=1 ;;
		*t*) local flag_t=1 ;;
		*n*) true ;;
	esac

	[ -z "$url" ] && error_echo "URL is not set"
	[ -z "$flag" ]
	# Get the real filename
	if [ -z "$fd" ] ; then
		filename="${url##*/}"
	fi

	# Make sure we can handle directory
	if [ -n "$fd" ]&&[ -d "$fd" ]||[ "$flag_t" ] ; then
		filename="${fd:-/tmp}/${filename}"
	fi

	if [ -n "$fd" ]&&[ ! -e "$fd" ] ; then
		filename="$fd"
	fi
	
	[ -d "$fd" ] && 

	if [ "$flag_dash" ] ; then
		{
			case "$url" in
				"https://"*|"http://"*|"ftp://"*|"tftp://"*) curl "$url" ;;
				"raw://"*) {
					url="${url#'raw://'}"
					nc "${url%:*}" "${url##*':'}"
					} ;;
				*) error_echo "Invaild URL Scheme"		
			esac 
		} | \
		{ if [ "$flag_dash" ] ; then
			cat
		elif [ ! "$flag_o" ]&&[ -e "$filename" ] ; then
			error_echo "File exists!"
		else
			cat > "$filename"
		fi }
	fi
}

