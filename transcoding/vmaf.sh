#!/bin/sh

ref="$1"
input="$2"
time nice -n 20 ffmpeg \
       	-i "${input}" -i "${ref}" -lavfi libvmaf=n_threads=32 -f null /dev/null
