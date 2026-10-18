#!/bin/sh
# Compute VMAF of a distorted video against its reference, handling resolution and frame-rate differences

usage() {
	echo "Usage: $0 [-m model] [-s WxH] [-l log.json] reference distorted" >&2
	echo "  -m  VMAF model version or libvmaf model spec (default: vmaf_4k_v0.6.1 above 1080p, else vmaf_v0.6.1)" >&2
	echo "  -s  evaluate at this resolution, scaling both inputs (default: reference resolution)" >&2
	echo "  -l  write per-frame scores to this JSON log" >&2
	exit 1
}

model=
size=
log=

while getopts "m:s:l:h" opt; do
	case "$opt" in
		m) model="$OPTARG" ;;
		s) size="$OPTARG" ;;
		l) log="$OPTARG" ;;
		*) usage ;;
	esac
done
shift $((OPTIND - 1))
[ $# -eq 2 ] || usage
ref="$1"
dist="$2"

probe() {
	ffprobe -v error -select_streams v:0 -show_entries stream="$2" -of csv=p=0 "$1"
}

# VMAF needs both inputs at the same size; upscaling the distorted video to the reference
# resolution with bicubic is the standard practice
[ -n "$size" ] || size="$(probe "$ref" width,height | tr , x)"
width="${size%x*}"
height="${size#*x}"

if [ -z "$model" ]; then
	[ "$height" -gt 1080 ] && model=vmaf_4k_v0.6.1 || model=vmaf_v0.6.1
fi
case "$model" in
	*=*) ;;
	*) model="version=$model" ;;
esac

# Compare the same instants: when the distorted video was frame-rate converted,
# resample the reference to its frame rate
ref_fps="$(probe "$ref" r_frame_rate)"
dist_fps="$(probe "$dist" r_frame_rate)"
ref_resample=
if [ "$(echo "$ref_fps $dist_fps" | awk '{ split($1, a, "/"); split($2, b, "/"); r = a[1] / a[2]; d = b[1] / b[2]; print ((r - d) / d > 0.001 || (d - r) / d > 0.001) }')" = 1 ]; then
	ref_resample=",fps=$dist_fps"
fi

echo "$0: evaluating at ${width}x${height}, model ${model#version=}${ref_resample:+, reference resampled to $dist_fps fps}" >&2

vmaf_opts="model=$model:n_threads=$(nproc):shortest=1:ts_sync_mode=nearest"
[ -n "$log" ] && vmaf_opts="$vmaf_opts:log_fmt=json:log_path='$log'"

nice -n 19 ffmpeg -hide_banner -nostdin \
	-i "$dist" -i "$ref" \
	-filter_complex "
		[0:v:0]setpts=PTS-STARTPTS,scale=$width:$height:flags=bicubic,format=yuv420p10le[dist];
		[1:v:0]setpts=PTS-STARTPTS$ref_resample,scale=$width:$height:flags=bicubic,format=yuv420p10le[ref];
		[dist][ref]libvmaf=$vmaf_opts" \
	-f null -
