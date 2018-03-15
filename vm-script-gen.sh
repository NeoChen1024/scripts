#!/bin/bash

SCRIPT_NAME="vm-script-gen"
AURTHOR="Neo_Chen <chenkolei@gmail.com>"
LICENSE="GPLv3 or Newer"

# usage() is important
usage()
{
cat <<EOF
Usage: $SCRIPT_NAME [-ah] [-n Name] [-c cputype] [-m mem] [-d file] [-s num] [-b bridge] [-e file] [-o file] [-v vga]

	-n	VM Name
	-v	Be Verbose
	-c	CPU type
	-m	Memory (in MiB)
	-d	Disk File (QCOW2 format)
	-s	Number of CPUs
	-b	Bridge Interface to use (If not specified, we use user network)
	-e	Enable EFI, and specify where the ROM is
	-o	Output to file
	-q	QEMU
	-a	Enable Audio
EOF
}

# Get command line aruguments

while getopts "hn:c:m:d:s:b:r:e:o:q:av" OPTION ; do
	case $OPTION in
	h)	usage
		exit 1;;
	o)	OUTFILE=$OPTARG;;
	v)	flag="${flag}"
		VGA="$OPTARG";;
	n)	flag="${flag}n"
		NAME="$OPTARG";;
	c)	flag="${flag}c"
		CPU="$OPTARG";;
	m)	flag="${flag}m"
		MEM="$OPTARG";;
	s)	flag="${flag}s"
		SMP="$OPTARG";;
	d)	flag="${flag}d"
		DISK="$OPTARG";;
	b)	flag="${flag}b"
		BRIDGE="$OPTARG";;
	e)	flag="${flag}e"
		EFI="$OPTARG";;
	q)	flag="${flag}q"
		QEMU="$OPTARG";;
	a)	flag="${flag}a";;
	esac
done

[[ "$flag" == *q* ]] && echo "nice -n 19 chrt -vi 0 ${QEMU} -accel kvm -balloon virtio \\" ||  echo "nice -n 19 chrt -vi 0 qemu-system-x86_64 -accel kvm -balloon virtio \\"
[[ "$flag" == *n* ]] && echo "-name ${NAME} \\"
[[ "$flag" == *c* ]] && echo "-cpu ${CPU} \\" || echo "-cpu host \\"
[[ "$flag" == *m* ]] && echo "-m ${MEM} \\" || echo "-m 512M \\"
[[ "$flag" == *s* ]] && echo "-smp ${SMP} \\"
[[ "$flag" == *d* ]] && echo "-blockdev driver=qcow2,node-name=hda,discard=unmap,cache.direct=on,detect-zeroes=unmap,file.driver=file,file.filename=\"${DISK}\" \\" && \
	echo '-device virtio-blk-pci,drive=hda \'
[[ "$flag" == *b* ]] && echo "-net nic,model=virtio -net bridge,br=${BRIDGE} \\"
[[ "$flag" == *e* ]] && echo "-pflash ${EFI} \\"
if [[ "$flag" == *v* ]] ;then
	if [[ "$VGA" == "virtio" ]] ; then
		echo '-device virtio-vga,virgl=on -display sdl,gl=on \'
	else
		echo "-vga ${VGA} \\"
	fi
fi
[[ "$flag" == *a* ]] && echo "-soundhw hda \\"
echo '"$@"'

