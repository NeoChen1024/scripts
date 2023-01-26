#!/bin/sh

ffmpeg \
	-i "$1" \
	-c:a libopus -b:a 256k \
	-c:v hevc -q:v 23 -preset veryslow -b:v 2048k -maxrate 3072k -g 60 -bufsize 8M \
	-c:s copy \
	"${1%.*}-small.mkv"
