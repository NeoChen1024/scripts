#!/bin/sh
# "HD-VCD": squeeze a video into a VCD-ish bitrate with two-pass VBR AV1 (SVT-AV1)

usage() {
	echo "Usage: $0 [-b bitrate] [-p preset] [-f film_grain] [-d] [-s height] [-r fps] [-o output] input" >&2
	echo "  -b  video target bitrate (default: 256k)" >&2
	echo "  -p  SVT-AV1 preset (default: 4)" >&2
	echo "  -f  film grain synthesis level, 0 = off (default: 0, typical: 8-12)" >&2
	echo "  -d  denoise source before film grain estimation" >&2
	echo "  -s  downscale to this height (default: keep source resolution)" >&2
	echo "  -r  convert to this frame rate, e.g. 30, 30000/1001, ntsc (default: keep source frame rate)" >&2
	echo "  -o  output file (default: <input without extension>-small.mkv)" >&2
	exit 1
}

bitrate=256k
preset=4
film_grain=0
film_grain_denoise=0
height=
fps=
output=

while getopts "b:p:f:ds:r:o:h" opt; do
	case "$opt" in
		b) bitrate="$OPTARG" ;;
		p) preset="$OPTARG" ;;
		f) film_grain="$OPTARG" ;;
		d) film_grain_denoise=1 ;;
		s) height="$OPTARG" ;;
		r) fps="$OPTARG" ;;
		o) output="$OPTARG" ;;
		*) usage ;;
	esac
done
shift $((OPTIND - 1))
[ -n "$1" ] || usage
[ -n "$output" ] || output="${1%.*}-small.mkv"

params="tune=0:keyint=10s:scd=1:enable-qm=1:enable-variance-boost=1:variance-octile=6:luminance-qp-bias=20:ac-bias=0.5"
params="$params:film-grain=$film_grain:film-grain-denoise=$film_grain_denoise"

vf="null"
[ -n "$fps" ] && vf="$vf,fps=$fps"
[ -n "$height" ] && vf="$vf,scale=-2:'min($height,ih)':flags=lanczos"

# Reserve room for Cues up front so the muxer needn't shift the whole file afterwards;
# ~64 bytes per second covers keyframe and subtitle cue points with plenty of headroom
duration="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" | cut -d. -f1)"
case "$duration" in
	''|*[!0-9]*) index_space=1048576 ;;
	*) index_space=$((duration * 64 + 65536)) ;;
esac

passlog="$(mktemp -u "${TMPDIR:-/tmp}/av1-2pass.XXXXXX")"
trap 'rm -f "$passlog"-*.log' EXIT

ffmpeg \
	-i "$1" \
	-an -sn \
	-vf "$vf" -pix_fmt yuv420p10le \
	-c:v libsvtav1 -preset "$preset" -b:v "$bitrate" -svtav1-params "$params" \
	-pass 1 -passlogfile "$passlog" \
	-f null - &&
ffmpeg \
	-i "$1" \
	-c:a libopus -b:a 64k -ac 2 \
	-vf "$vf" -pix_fmt yuv420p10le \
	-c:v libsvtav1 -preset "$preset" -b:v "$bitrate" -svtav1-params "$params" \
	-pass 2 -passlogfile "$passlog" \
	-c:s srt \
	-reserve_index_space "$index_space" \
	"$output"
