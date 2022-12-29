#!/bin/sh

ffmpeg -init_hw_device qsv=hw -filter_hw_device hw \
	-i "$1" \
	-c:a libopus -b:a 64k \
	-vf 'hwupload=extra_hw_frames=64,vpp_qsv=denoise=100' -c:v hevc_qsv -preset veryslow -b:v 384k -maxrate 384k -g 60 -bufsize 2M \
	-c:s copy \
	"${1%.*}-small.mkv"
