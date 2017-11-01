#!/bin/bash

FPATH="./www/"
TMPFILE="/tmp/sed-$UID.tmp"

for i in $(find "$FPATH" -name \*.html) ; do
	if grep 'href="/neostyle.css"' "$i" ; then
	sed 's|href="/neostyle.css"|href="/~neo_chen/neostyle.css"|' "$i" > "$TMPFILE"
	cp -v "$TMPFILE" "$i"
	fi
done
