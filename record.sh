#!/bin/sh

ffmpeg -init_hw_device qsv=hw -filter_hw_device hw \
	-c:v mjpeg_qsv -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 30 -thread_queue_size 1024 -i /dev/video0 \
	-f pulse -thread_queue_size 1024 -i alsa_input.usb-Jieli_Technology_USB_PHY_2.0-02.mono-fallback \
	-c:a libopus -b:a 64k \
	-vf 'hwupload=extra_hw_frames=64,vpp_qsv=denoise=100' -c:v hevc_qsv -preset veryslow -b:v 384k -maxrate 384k -g 60 -bufsize 2M \
	$(date +'%Y%m%dT%H%M%S').mkv
