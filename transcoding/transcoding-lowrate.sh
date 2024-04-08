#!/bin/sh

ffmpeg \
	-i "$1" \
	-c:a libopus -b:a 64k \
	-c:v hevc -q:v 23 -preset veryslow -profile:v main -tier:v high -level:v 6.2 -b:v 256k -maxrate 1024k -bufsize 8M \
	-c:s srt \
	"${1%.*}-small.mkv"
