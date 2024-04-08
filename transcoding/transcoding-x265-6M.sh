#!/bin/sh

time ffmpeg \
	-i "$1" \
	-c:a libopus -b:a 256k \
	-c:v hevc -q:v 20 -preset slow -tier:v high -level:v 6.2 -b:v 6000k -maxrate 8192k -bufsize 64M \
	-c:s copy \
	"${2}"
