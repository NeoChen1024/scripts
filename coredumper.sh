#!/usr/bin/env bash

# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
# 
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 
# For more information, please refer to <http://unlicense.org/>

# sysctl config:
# kernel/core_pattern=|/some/place/coredumper %e %P %u %g %s %t %E

SCRIPT_NAME="coredumper"
AURTHOR="Neo_Chen <chenkolei@gmail.com>"
LICENSE="Public Domain"

if [ "$EUID" != 0 ]||[ "$UID" != 0 ] ;then
	echo "This script should not be run by any user!!"
	exit 200
fi

name="$1"
pid="$2"
uid="$3"
gid="$4"
signal="$5"
epoch="$6"
path="$7"

# Echo the message into Kernel Message Buffer
echo "coredump: [PATH=${path//'!'/"/"}, PID=${pid}, ID=${uid}:${gid}, SIG=${signal}]" > /dev/kmsg

# Compress the coredump with gzip
# because all *NIX Systems have it
gzip -c > /var/coredump/"${name}:${pid}:${uid}:${gid}:${signal}:${epoch}.coredump.gz"
## zstd -19c -T0 -c > /var/coredump/"${name}:${pid}:${uid}:${gid}:${signal}:${epoch}.coredump.zst"

# vim: set tabstop=8:softtabstop=8:shiftwidth=8
