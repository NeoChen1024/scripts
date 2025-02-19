#!/bin/sh

time nice -n 20 ffmpeg \
	-i "$1" \
	-c:a libopus -b:a 256k \
	-c:v hevc_nvenc -gpu 0 -rc-lookahead 32 -multipass fullres -profile:v main -preset:v p7 -tier:v high -level:v 6.2 -b:v 6000k -maxrate 12000k -bufsize 16M \
	-c:s copy \
	"${2}"
