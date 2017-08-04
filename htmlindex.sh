#!/bin/bash
# Static HTML File Index Generator
# Licensed under WTFPL

dir="$1"
title="$2"
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
		<h1>${title}</h>
EOF

cd "${1:-.}"
for i in * ; do
	echo -e "\t\t<a href=\"$i\">$i</a>"
done

cat <<EOF
	</body>
</html>
EOF

