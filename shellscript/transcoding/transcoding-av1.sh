#!/bin/sh

ffmpeg \
	-i "$1" \
	-c:a libopus -b:a 64k \
	-c:v librav1e -q:v 25 -rav1e-params speed=0 -b:v 256k -maxrate 1024k -bufsize 8M \
	-c:s srt \
	"${1%.*}-small.mkv"
