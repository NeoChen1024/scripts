#!/bin/bash
# Static HTML File Index Generator
# Licensed under GNU GPLv3 or newer

dir="$1"
title="${2:-$PWD}"
oldpwd="$PWD"

cat <<EOF
<!DOCTYPE html>
<html>
	<head>
		<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>${title}</title>
	</head>
	<body>
		<h1>${title}</h1>
		<table>
		<tr><th>Filename</th>	<th>Size</th><tr>
EOF

cd "${1:-.}"
for i in * ; do
	echo -e "\t\t<tr><td><a href=\"$i\">$i</a></td><td>`wc -c "$i" | cut -f 1 -d \ `</td></tr>"
done

cat <<EOF
	</table>
	</body>
</html>
EOF

