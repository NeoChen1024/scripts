#!/bin/bash
set -v
install -m0755 uptime.sh /usr/bin/uptimerec
install -m0644 uptimerec.service /lib/systemd/system/uptimerec.service
