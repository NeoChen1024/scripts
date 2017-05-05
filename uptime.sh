#!/bin/bash
# Uptime Recorder v0.9
# 2017/04/29
# Neo_Chen <chenkolei@gmail.com>

dir="$1"
uptime=$(cut -d \  -f 1 /proc/uptime)
idletime=$(cut -d \  -f 2 /proc/uptime)
current="$(date +'AD %Y/%m/%d [%^a] %H:%M:%S(%s)')"
since="$(date -d @$(sed -n '/^btime /s///p' /proc/stat) +'AD %Y/%m/%d [%^a] %H:%M:%S(%s)')"
if [ -d "$dir"  ];then
	echo -en "$since\t$current\t$uptime\t$idletime\n" >> "$dir"/uptimerec.lst
	echo -en "$since\t$current\t$uptime\t$idletime\n" >> "$dir"/stat.lst
	cat /proc/stat >> "$dir"/stat.lst
	echo -e "\n%%%%----%%%%\n" >> "$dir"/stat.lst
else
	exit 1
fi

if [ -e "$dir"/totaluptime.rec ];then
	last=$(cat "$dir"/totaluptime.rec)
	echo "scale=3;${last}+${uptime}" | bc -l > "$dir"/totaluptime.rec
else
	echo $uptime > "$dir"/totaluptime.rec
fi


